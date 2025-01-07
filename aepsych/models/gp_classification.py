#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import gpytorch
import numpy as np
import torch
from aepsych.config import Config
from aepsych.factory.default import default_mean_covar_factory
from aepsych.models.base import AEPsychModelDeviceMixin
from aepsych.models.inducing_points import GreedyVarianceReduction
from aepsych.models.inducing_points.base import InducingPointAllocator
from aepsych.utils import get_dims, get_optimizer_options, promote_0d
from aepsych.utils_logging import getLogger
from gpytorch.likelihoods import BernoulliLikelihood, BetaLikelihood, Likelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from scipy.special import owens_t
from scipy.stats import norm
from torch.distributions import Normal

logger = getLogger()


class GPClassificationModel(AEPsychModelDeviceMixin, ApproximateGP):
    """Probit-GP model with variational inference.

    From a conventional ML perspective this is a GP Classification model,
    though in the psychophysics context it can also be thought of as a
    nonlinear generalization of the standard linear model for 1AFC or
    yes/no trials.

    For more on variational inference, see e.g.
    https://docs.gpytorch.ai/en/v1.1.1/examples/04_Variational_and_Approximate_GPs/
    """

    _batch_size = 1
    _num_outputs = 1
    stimuli_per_trial = 1
    outcome_type = "binary"

    def __init__(
        self,
        dim: int,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        inducing_point_method: Optional[InducingPointAllocator] = None,
        inducing_size: int = 100,
        max_fit_time: Optional[float] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the GP Classification model

        Args:
            dim (int): The number of dimensions in the parameter space.
            mean_module (gpytorch.means.Mean, optional): GP mean class. Defaults to a constant with a normal prior.
            covar_module (gpytorch.kernels.Kernel, optional): GP covariance kernel class. Defaults to scaled RBF with a
                gamma prior.
            likelihood (gpytorch.likelihood.Likelihood, optional): The likelihood function to use. If None defaults to
                Bernouli likelihood.
            inducing_point_method (InducingPointAllocator, optional): The method to use for selecting inducing points.
                If not set, a GreedyVarianceReduction is made.
            inducing_size (int): Number of inducing points. Defaults to 100.
            max_fit_time (float, optional): The maximum amount of time, in seconds, to spend fitting the model. If None,
                there is no limit to the fitting time.
            optimizer_options (Dict[str, Any], optional): Optimizer options to pass to the SciPy optimizer during
                fitting. Assumes we are using L-BFGS-B.
        """
        self.dim = dim
        self.max_fit_time = max_fit_time
        self.inducing_size = inducing_size

        self.optimizer_options = (
            {"options": optimizer_options} if optimizer_options else {"options": {}}
        )

        if likelihood is None:
            likelihood = BernoulliLikelihood()

        if mean_module is None or covar_module is None:
            default_mean, default_covar = default_mean_covar_factory(
                dim=self.dim, stimuli_per_trial=self.stimuli_per_trial
            )

        self.inducing_point_method = inducing_point_method or GreedyVarianceReduction(
            dim=self.dim
        )
        inducing_points = self.inducing_point_method.allocate_inducing_points(
            num_inducing=self.inducing_size,
            covar_module=covar_module or default_covar,
        )

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=torch.Size([self._batch_size])
        ).to(inducing_points)

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        )
        super().__init__(variational_strategy)

        self.likelihood = likelihood
        self.mean_module = mean_module or default_mean
        self.covar_module = covar_module or default_covar

        self._fresh_state_dict = deepcopy(self.state_dict())
        self._fresh_likelihood_dict = deepcopy(self.likelihood.state_dict())

    @classmethod
    def from_config(cls, config: Config) -> GPClassificationModel:
        """Alternate constructor for GPClassification model from a configuration.

        This is used when we recursively build a full sampling strategy
        from a configuration. TODO: document how this works in some tutorial.

        Args:
            config (Config): A configuration containing keys/values matching this class

        Returns:
            GPClassificationModel: Configured class instance.
        """

        classname = cls.__name__
        inducing_size = config.getint(classname, "inducing_size", fallback=100)

        dim = config.getint(classname, "dim", fallback=None)
        if dim is None:
            dim = get_dims(config)

        mean_covar_factory = config.getobj(
            classname, "mean_covar_factory", fallback=default_mean_covar_factory
        )

        mean, covar = mean_covar_factory(config)
        max_fit_time = config.getfloat(classname, "max_fit_time", fallback=None)

        inducing_point_method_class = config.getobj(
            classname, "inducing_point_method", fallback=GreedyVarianceReduction
        )
        # Check if allocator class has a `from_config` method
        if hasattr(inducing_point_method_class, "from_config"):
            inducing_point_method = inducing_point_method_class.from_config(config)
        else:
            inducing_point_method = inducing_point_method_class()

        likelihood_cls = config.getobj(classname, "likelihood", fallback=None)

        if likelihood_cls is not None:
            if hasattr(likelihood_cls, "from_config"):
                likelihood = likelihood_cls.from_config(config)
            else:
                likelihood = likelihood_cls()
        else:
            likelihood = None  # fall back to __init__ default

        optimizer_options = get_optimizer_options(config, classname)

        return cls(
            dim=dim,
            inducing_size=inducing_size,
            mean_module=mean,
            covar_module=covar,
            max_fit_time=max_fit_time,
            inducing_point_method=inducing_point_method,
            likelihood=likelihood,
            optimizer_options=optimizer_options,
        )

    def _reset_hyperparameters(self) -> None:
        """Reset hyperparameters to their initial values."""
        # warmstart_hyperparams affects hyperparams but not the variational strat,
        # so we keep the old variational strat (which is only refreshed
        # if warmstart_induc=False).
        vsd = self.variational_strategy.state_dict()  # type: ignore
        vsd_hack = {f"variational_strategy.{k}": v for k, v in vsd.items()}
        state_dict = deepcopy(self._fresh_state_dict)
        state_dict.update(vsd_hack)
        self.load_state_dict(state_dict)
        self.likelihood.load_state_dict(self._fresh_likelihood_dict)

    def _reset_variational_strategy(self) -> None:
        if self.train_inputs is not None:
            # remember original device
            device = self.device
            inducing_points = self.inducing_point_method.allocate_inducing_points(
                num_inducing=self.inducing_size,
                covar_module=self.covar_module,
                inputs=self.train_inputs[0],
            ).to(device)
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(0), batch_shape=torch.Size([self._batch_size])
            ).to(device)
            self.variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ).to(device)

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        warmstart_hyperparams: bool = False,
        warmstart_induc: bool = False,
        **kwargs,
    ) -> None:
        """Fit underlying model.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.LongTensor): Responses.
            warmstart_hyperparams (bool): Whether to reuse the previous hyperparameters (True) or fit from scratch
                (False). Defaults to False.
            warmstart_induc (bool): Whether to reuse the previous inducing points or fit from scratch (False).
                Defaults to False.
        """
        self.set_train_data(train_x, train_y)

        # by default we reuse the model state and likelihood. If we
        # want a fresh fit (no warm start), copy the state from class initialization.
        if not warmstart_hyperparams:
            self._reset_hyperparameters()

        if not warmstart_induc or (
            self.inducing_point_method.last_allocator_used is None
        ):
            self._reset_variational_strategy()

        n = train_y.shape[0]
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, n)

        if "optimizer_kwargs" in kwargs:
            self._fit_mll(mll, **kwargs)
        else:
            self._fit_mll(mll, optimizer_kwargs=self.optimizer_options, **kwargs)

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample from underlying model.

        Args:
            x (torch.Tensor): Points at which to sample.
            num_samples (int): Number of samples to return.
            kwargs are ignored

        Returns:
            torch.Tensor: Posterior samples [num_samples x dim]
        """
        x = x.to(self.device)
        return self.posterior(x).rsample(torch.Size([num_samples])).detach().squeeze()

    def predict(
        self, x: torch.Tensor, probability_space: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            probability_space (bool): Return outputs in units of
                response probability instead of latent function value. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at queries points.
        """
        with torch.no_grad():
            x = x.to(self.device)
            post = self.posterior(x)
            fmean = post.mean.squeeze()
            fvar = post.variance.squeeze()
        if probability_space:
            if isinstance(self.likelihood, BernoulliLikelihood):
                # Probability-space mean and variance for Bernoulli-probit models is
                # available in closed form, Proposition 1 in Letham et al. 2022 (AISTATS).
                a_star = fmean / torch.sqrt(1 + fvar)
                pmean = Normal(0, 1).cdf(a_star)
                t_term = torch.tensor(
                    owens_t(
                        a_star.cpu().numpy(), 1 / np.sqrt(1 + 2 * fvar.cpu().numpy())
                    ),
                    dtype=a_star.dtype,
                ).to(self.device)
                pvar = pmean - 2 * t_term - pmean.square()
                return promote_0d(pmean), promote_0d(pvar)
            else:
                fsamps = post.sample(torch.Size([10000]))
                if hasattr(self.likelihood, "objective"):
                    psamps = self.likelihood.objective(fsamps)
                else:
                    psamps = norm.cdf(fsamps)
                pmean, pvar = psamps.mean(0), psamps.var(0)
                return promote_0d(pmean), promote_0d(pvar)

        else:
            return promote_0d(fmean), promote_0d(fvar)

    def predict_probability(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance in probability space.

        Args:
            x (torch.Tensor): Points at which to predict from the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at queries points.
        """
        return self.predict(x, probability_space=True)

    def update(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs):
        """Perform a warm-start update of the model from previous fit.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.Tensor): Responses.
        """
        return self.fit(
            train_x, train_y, warmstart_hyperparams=True, warmstart_induc=True, **kwargs
        )


class GPBetaRegressionModel(GPClassificationModel):
    outcome_type = "percentage"

    def __init__(
        self,
        dim: int,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        inducing_point_method: Optional[InducingPointAllocator] = None,
        inducing_size: int = 100,
        max_fit_time: Optional[float] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the GP Beta Regression model

        Args:
            dim (int): The number of dimensions in the parameter space.
            mean_module (gpytorch.means.Mean, optional): GP mean class. Defaults to a constant with a normal prior. Defaults to None.
            covar_module (gpytorch.kernels.Kernel, optional): GP covariance kernel class. Defaults to scaled RBF with a
                gamma prior.
            likelihood (gpytorch.likelihood.Likelihood, optional): The likelihood function to use. If None defaults to
                Beta likelihood.
            inducing_point_method (InducingPointAllocator, optional): The method to use for selecting inducing points.
                If not set, a GreedyVarianceReduction is made.
            inducing_size (int): Number of inducing points. Defaults to 100.
            max_fit_time (float, optional): The maximum amount of time, in seconds, to spend fitting the model. If None,
                there is no limit to the fitting time. Defaults to None.
        """
        if likelihood is None:
            likelihood = BetaLikelihood()

        super().__init__(
            dim=dim,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            inducing_size=inducing_size,
            max_fit_time=max_fit_time,
            inducing_point_method=inducing_point_method,
            optimizer_options=optimizer_options,
        )
