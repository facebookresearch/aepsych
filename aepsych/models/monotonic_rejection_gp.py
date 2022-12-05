#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gpytorch
import numpy as np
import torch
from aepsych.acquisition.rejection_sampler import RejectionSampler
from aepsych.config import Config
from aepsych.factory.factory import monotonic_mean_covar_factory
from aepsych.kernels.rbf_partial_grad import RBFKernelPartialObsGrad
from aepsych.means.constant_partial_grad import ConstantMeanPartialObsGrad
from aepsych.models.base import AEPsychMixin
from aepsych.models.utils import select_inducing_points
from aepsych.utils import _process_bounds, promote_0d
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood, Likelihood
from gpytorch.means import Mean
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from scipy.stats import norm
from torch import Tensor


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
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        dim: Optional[int] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        fixed_prior_mean: Optional[float] = None,
        num_induc: int = 25,
        num_samples: int = 250,
        num_rejection_samples: int = 5000,
        inducing_point_method: str = "auto",
    ) -> None:
        """Initialize MonotonicRejectionGP.

        Args:
            likelihood (str): Link function and likelihood. Can be 'probit-bernoulli' or
                'identity-gaussian'.
            monotonic_idxs (List[int]): List of which columns of x should be given monotonicity
            constraints.
            fixed_prior_mean (Optional[float], optional): Fixed prior mean. If classification, should be the prior
            classification probability (not the latent function value). Defaults to None.
            covar_module (Optional[Kernel], optional): Covariance kernel to use (default: scaled RBF).
            mean_module (Optional[Mean], optional): Mean module to use (default: constant mean).
            num_induc (int, optional): Number of inducing points for variational GP.]. Defaults to 25.
            num_samples (int, optional): Number of samples for estimating posterior on preDict or
            acquisition function evaluation. Defaults to 250.
            num_rejection_samples (int, optional): Number of samples used for rejection sampling. Defaults to 4096.
            acqf (MonotonicMCAcquisition, optional): Acquisition function to use for querying points. Defaults to MonotonicMCLSE.
            objective (Optional[MCAcquisitionObjective], optional): Transformation of GP to apply before computing acquisition function. Defaults to identity transform for gaussian likelihood, probit transform for probit-bernoulli.
            extra_acqf_args (Optional[Dict[str, object]], optional): Additional arguments to pass into the acquisition function. Defaults to None.
        """
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        if likelihood is None:
            likelihood = BernoulliLikelihood()

        self.inducing_size = num_induc
        self.inducing_point_method = inducing_point_method
        inducing_points = select_inducing_points(
            inducing_size=self.inducing_size,
            bounds=self.bounds,
            method="sobol",
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

    def fit(self, train_x: Tensor, train_y: Tensor, **kwargs) -> None:
        """Fit the model

        Args:
            train_x (Tensor): Training x points
            train_y (Tensor): Training y points. Should be (n x 1).
        """
        self.set_train_data(train_x, train_y)

        self.inducing_points = select_inducing_points(
            inducing_size=self.inducing_size,
            model=self,
            X=self.train_inputs[0],
            bounds=self.bounds,
            method=self.inducing_point_method,
        )
        self._set_model(train_x, train_y)

    def _set_model(
        self,
        train_x: Tensor,
        train_y: Tensor,
        model_state_dict: Optional[Dict[str, Tensor]] = None,
        likelihood_state_dict: Optional[Dict[str, Tensor]] = None,
    ) -> None:
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
        mll = fit_gpytorch_mll(mll)

    def update(self, train_x: Tensor, train_y: Tensor, warmstart: bool = True) -> None:
        """
        Update the model with new data.

        Expects the full set of data, not the incremental new data.

        Args:
            train_x (Tensor): Train X.
            train_y (Tensor): Train Y. Should be (n x 1).
            warmstart (bool): If True, warm-start model fitting with current parameters.
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
        x: Tensor,
        num_samples: Optional[int] = None,
        num_rejection_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample from monotonic GP

        Args:
            x (Tensor): tensor of n points at which to sample
            num_samples (int, optional): how many points to sample (default: self.num_samples)

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
        self, x: Tensor, probability_space: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Predict

        Args:
            x: tensor of n points at which to predict.

        Returns: tuple (f, var) where f is (n,) and var is (n,)
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

    def _augment_with_deriv_index(self, x: Tensor, indx):
        return torch.cat(
            (x, indx * torch.ones(x.shape[0], 1)),
            dim=1,
        )

    def _get_deriv_constraint_points(self):
        deriv_cp = torch.tensor([])
        for i in self.monotonic_idxs:
            induc_i = self._augment_with_deriv_index(self.inducing_points, i + 1)
            deriv_cp = torch.cat((deriv_cp, induc_i), dim=0)
        return deriv_cp

    @classmethod
    def from_config(cls, config: Config) -> MonotonicRejectionGP:
        classname = cls.__name__
        num_induc = config.gettensor(classname, "num_induc", fallback=25)
        num_samples = config.gettensor(classname, "num_samples", fallback=250)
        num_rejection_samples = config.getint(
            classname, "num_rejection_samples", fallback=5000
        )

        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = config.getint(classname, "dim", fallback=None)

        mean_covar_factory = config.getobj(
            classname, "mean_covar_factory", fallback=monotonic_mean_covar_factory
        )

        mean, covar = mean_covar_factory(config)

        monotonic_idxs: List[int] = config.getlist(
            classname, "monotonic_idxs", fallback=[-1]
        )

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
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Evaluate GP

        Args:
            x (torch.Tensor): Tensor of points at which GP should be evaluated.

        Returns:
            gpytorch.distributions.MultivariateNormal: Distribution object
                holding mean and covariance at x.
        """

        # final dim is deriv index, we only normalize the "real" dims
        transformed_x = x.clone()
        transformed_x[..., :-1] = self.normalize_inputs(transformed_x[..., :-1])
        mean_x = self.mean_module(transformed_x)
        covar_x = self.covar_module(transformed_x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
