#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gpytorch
import torch
from aepsych.acquisition.rejection_sampler import RejectionSampler
from aepsych.config import Config
from aepsych.factory.monotonic import monotonic_mean_covar_factory
from aepsych.kernels.rbf_partial_grad import RBFKernelPartialObsGrad
from aepsych.means.constant_partial_grad import ConstantMeanPartialObsGrad
from aepsych.models.base import AEPsychMixin
from aepsych.models.inducing_points import GreedyVarianceReduction
from aepsych.models.inducing_points.base import InducingPointAllocator
from aepsych.utils import _process_bounds, get_dims, get_optimizer_options, promote_0d
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood, Likelihood
from gpytorch.means import Mean
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from scipy.stats import norm


class MonotonicRejectionGP(AEPsychMixin, ApproximateGP):
    """A monotonic GP using rejection sampling.

    This takes the same insight as in e.g. Riihimäki & Vehtari 2010 (that the derivative of a GP
    is likewise a GP) but instead of approximately optimizing the likelihood of the model
    using EP, we optimize an unconstrained model by VI and then draw monotonic samples
    by rejection sampling.

    References:
        Riihimäki, J., & Vehtari, A. (2010). Gaussian processes with monotonicity information.
            Journal of Machine Learning Research, 9, 645–652.
    """

    _num_outputs = 1
    stimuli_per_trial = 1
    outcome_type = "binary"

    def __init__(
        self,
        monotonic_idxs: Sequence[int],
        lb: torch.Tensor,
        ub: torch.Tensor,
        dim: Optional[int] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        fixed_prior_mean: Optional[float] = None,
        num_induc: int = 25,
        num_samples: int = 250,
        num_rejection_samples: int = 5000,
        inducing_point_method: Optional[InducingPointAllocator] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize MonotonicRejectionGP.

        Args:
            monotonic_idxs (Sequence[int]): List of which columns of x should be given monotonicity
            constraints.
            lb (torch.Tensor): Lower bounds of the parameters.
            ub (torch.Tensor): Upper bounds of the parameters.
            dim (int, optional): The number of dimensions in the parameter space. If None, it is inferred from the size.
            covar_module (Kernel, optional): Covariance kernel to use. Default is scaled RBF.
            mean_module (Mean, optional): Mean module to use. Default is constant mean.
            likelihood (str, optional): Link function and likelihood. Can be 'probit-bernoulli' or
                'identity-gaussian'.
            fixed_prior_mean (float, optional): Fixed prior mean. If classification, should be the prior
            classification probability (not the latent function value). Defaults to None.
            num_induc (int): Number of inducing points for variational GP.]. Defaults to 25.
            num_samples (int): Number of samples for estimating posterior on preDict or
            acquisition function evaluation. Defaults to 250.
            num_rejection_samples (int): Number of samples used for rejection sampling. Defaults to 4096.
            inducing_point_method (InducingPointAllocator, optional): Method for selecting inducing points. If not set,
                a GreedyVarianceReduction is created.
            optimizer_options (Dict[str, Any], optional): Optimizer options to pass to the SciPy optimizer during
                fitting. Assumes we are using L-BFGS-B.
        """
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        if likelihood is None:
            likelihood = BernoulliLikelihood()

        self.inducing_size = num_induc
        self.inducing_point_method = inducing_point_method or GreedyVarianceReduction(
            dim=self.dim
        )

        inducing_points = self.inducing_point_method.allocate_inducing_points(
            num_inducing=self.inducing_size
        )

        inducing_points_aug = self._augment_with_deriv_index(inducing_points, 0)
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points_aug.size(0)
        )
        variational_strategy = VariationalStrategy(
            model=self,
            inducing_points=inducing_points_aug,
            variational_distribution=variational_distribution,
            learn_inducing_locations=False,
        )

        if mean_module is None:
            mean_module = ConstantMeanPartialObsGrad()

        if fixed_prior_mean is not None:
            if isinstance(likelihood, BernoulliLikelihood):
                fixed_prior_mean = norm.ppf(fixed_prior_mean)
            mean_module.constant.requires_grad_(False)
            mean_module.constant.copy_(torch.tensor(fixed_prior_mean))

        if covar_module is None:
            ls_prior = gpytorch.priors.GammaPrior(
                concentration=4.6, rate=1.0, transform=lambda x: 1 / x
            )
            ls_prior_mode = ls_prior.rate / (ls_prior.concentration + 1)
            ls_constraint = gpytorch.constraints.GreaterThan(
                lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
            )
            covar_module = gpytorch.kernels.ScaleKernel(
                RBFKernelPartialObsGrad(
                    lengthscale_prior=ls_prior,
                    lengthscale_constraint=ls_constraint,
                    ard_num_dims=dim,
                ),
                outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
            )

        super().__init__(variational_strategy)

        self.bounds_ = torch.stack([self.lb, self.ub])
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.likelihood = likelihood

        self.num_induc = num_induc
        self.monotonic_idxs = monotonic_idxs
        self.num_samples = num_samples
        self.num_rejection_samples = num_rejection_samples
        self.fixed_prior_mean = fixed_prior_mean
        self.inducing_points = inducing_points
        self.optimizer_options = (
            {"options": optimizer_options} if optimizer_options else {"options": {}}
        )

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs) -> None:
        """Fit the model

        Args:
            train_x (torch.Tensor): Training x points
            train_y (torch.Tensor): Training y points. Should be (n x 1).
        """
        self.set_train_data(train_x, train_y)

        self.inducing_points = self.inducing_point_method.allocate_inducing_points(
            num_inducing=self.inducing_size,
            covar_module=self.covar_module,
            inputs=self._augment_with_deriv_index(self.train_inputs[0], 0),
        )
        self._set_model(train_x, train_y)

    def _set_model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        model_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        likelihood_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Sets the model with the given data and state dicts.

        Args:
            train_x (torch.Tensor): Training x points
            train_y (torch.Tensor): Training y points. Should be (n x 1).
            model_state_dict (Dict[str, torch.Tensor], optional): State dict for the model
            likelihood_state_dict (Dict[str, torch.Tensor], optional): State dict for the likelihood
        """
        train_x_aug = self._augment_with_deriv_index(train_x, 0)
        self.set_train_data(train_x_aug, train_y)
        # Set model parameters
        if model_state_dict is not None:
            self.load_state_dict(model_state_dict)
        if likelihood_state_dict is not None:
            self.likelihood.load_state_dict(likelihood_state_dict)

        # Fit!
        mll = VariationalELBO(
            likelihood=self.likelihood, model=self, num_data=train_y.numel()
        )
        mll = fit_gpytorch_mll(mll, optimizer_kwargs=self.optimizer_options)

    def update(
        self, train_x: torch.Tensor, train_y: torch.Tensor, warmstart: bool = True
    ) -> None:
        """
        Update the model with new data.

        Expects the full set of data, not the incremental new data.

        Args:
            train_x (torch.Tensor): Train X.
            train_y (torch.Tensor): Train Y. Should be (n x 1).
            warmstart (bool): If True, warm-start model fitting with current parameters. Defaults to True.
        """
        if warmstart:
            model_state_dict = self.state_dict()
            likelihood_state_dict = self.likelihood.state_dict()
        else:
            model_state_dict = None
            likelihood_state_dict = None
        self._set_model(
            train_x=train_x,
            train_y=train_y,
            model_state_dict=model_state_dict,
            likelihood_state_dict=likelihood_state_dict,
        )

    def sample(
        self,
        x: torch.Tensor,
        num_samples: Optional[int] = None,
        num_rejection_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample from monotonic GP

        Args:
            x (torch.Tensor): tensor of n points at which to sample
            num_samples (int, optional): how many points to sample. Default is self.num_samples.
            num_rejection_samples (int): how many samples to use for rejection sampling. Default is self.num_rejection_samples.

        Returns: a Tensor of shape [n_samp, n]
        """
        if num_samples is None:
            num_samples = self.num_samples
        if num_rejection_samples is None:
            num_rejection_samples = self.num_rejection_samples

        rejection_ratio = 20
        if num_samples * rejection_ratio > num_rejection_samples:
            warnings.warn(
                f"num_rejection_samples should be at least {rejection_ratio} times greater than num_samples."
            )

        n = x.shape[0]
        # Augment with derivative index
        x_aug = self._augment_with_deriv_index(x, 0)
        # Add in monotonicity constraint points
        deriv_cp = self._get_deriv_constraint_points()
        x_aug = torch.cat((x_aug, deriv_cp), dim=0)
        assert x_aug.shape[0] == x.shape[0] + len(
            self.monotonic_idxs * self.inducing_points.shape[0]
        )
        constrained_idx = torch.arange(n, x_aug.shape[0])

        with torch.no_grad():
            posterior = self.posterior(x_aug)
        sampler = RejectionSampler(
            num_samples=num_samples,
            num_rejection_samples=num_rejection_samples,
            constrained_idx=constrained_idx,
        )
        samples = sampler(posterior)
        samples_f = samples[:, :n, 0].detach().cpu()
        return samples_f

    def predict(
        self, x: torch.Tensor, probability_space: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict

        Args:
            x (torch.Tensor): tensor of n points at which to predict.
            probability_space (bool): whether to return in probability space. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at query points.
        """
        samples_f = self.sample(x)
        mean = torch.mean(samples_f, dim=0).squeeze()
        variance = torch.var(samples_f, dim=0).clamp_min(0).squeeze()

        if probability_space:
            return (
                torch.Tensor(promote_0d(norm.cdf(mean))),
                torch.Tensor(promote_0d(norm.cdf(variance))),
            )

        return mean, variance

    def predict_probability(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict in probability space

        Args:
            x (torch.Tensor): Points at which to predict.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at query points.
        """
        return self.predict(x, probability_space=True)

    def _augment_with_deriv_index(self, x: torch.Tensor, indx: int) -> torch.Tensor:
        """Augment input with derivative index

        Args:
            x (torch.Tensor): Input tensor
            indx (int): Derivative index

        Returns:
            torch.Tensor: Augmented tensor
        """
        return torch.cat(
            (x, indx * torch.ones(x.shape[0], 1)),
            dim=1,
        )

    def _get_deriv_constraint_points(self) -> torch.Tensor:
        """Get derivative constraint points"""
        deriv_cp = torch.tensor([])
        for i in self.monotonic_idxs:
            induc_i = self._augment_with_deriv_index(self.inducing_points, i + 1)
            deriv_cp = torch.cat((deriv_cp, induc_i), dim=0)
        return deriv_cp

    @classmethod
    def from_config(cls, config: Config) -> MonotonicRejectionGP:
        """Alternate constructor for MonotonicRejectionGP

        Args:
            config (Config): a configuration containing keys/values matching this class

        Returns:
            MonotonicRejectionGP: configured class instance
        """
        classname = cls.__name__
        num_induc = config.gettensor(classname, "num_induc", fallback=25)
        num_samples = config.gettensor(classname, "num_samples", fallback=250)
        num_rejection_samples = config.getint(
            classname, "num_rejection_samples", fallback=5000
        )

        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = get_dims(config)

        mean_covar_factory = config.getobj(
            classname, "mean_covar_factory", fallback=monotonic_mean_covar_factory
        )

        mean, covar = mean_covar_factory(config)

        monotonic_idxs: List[int] = config.getlist(
            classname, "monotonic_idxs", fallback=[-1]
        )

        optimizer_options = get_optimizer_options(config, classname)

        return cls(
            monotonic_idxs=monotonic_idxs,
            lb=lb,
            ub=ub,
            dim=dim,
            num_induc=num_induc,
            num_samples=num_samples,
            num_rejection_samples=num_rejection_samples,
            mean_module=mean,
            covar_module=covar,
            optimizer_options=optimizer_options,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Evaluate GP

        Args:
            x (torch.Tensor): Tensor of points at which GP should be evaluated.

        Returns:
            gpytorch.distributions.MultivariateNormal: Distribution object
                holding mean and covariance at x.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
