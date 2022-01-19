#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time
from typing import Optional, Tuple, Union

import gpytorch
import numpy as np
import torch
from aepsych.config import Config
from aepsych.factory.factory import default_mean_covar_factory
from aepsych.models.base import AEPsychMixin
from aepsych.utils import _process_bounds, make_scaled_sobol
from aepsych.utils_logging import getLogger
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.likelihoods import BernoulliLikelihood, Likelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import MeanFieldVariationalDistribution, VariationalStrategy

logger = getLogger()


class GPClassificationModel(AEPsychMixin, ApproximateGP, GPyTorchModel):
    """Probit-GP model with variational inference.

    From a conventional ML perspective this is a GP Classification model,
    though in the psychophysics context it can also be thought of as a
    nonlinear generalization of the standard linear model for 1AFC or
    yes/no trials.

    For more on variational inference, see e.g.
    https://docs.gpytorch.ai/en/v1.1.1/examples/04_Variational_and_Approximate_GPs/
    """

    _num_outputs = 1
    outcome_type = "single_probit"

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        dim: Optional[int] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        inducing_size: int = 10,
        max_fit_time: Optional[float] = None,
    ):
        """Initialize the GP Classification model

        Args:
            inducing_size (int, optional): Number of inducing points. Defaults to 10.
            mean_module (gpytorch.means.Mean, optional): GP mean class. Defaults
                to a constant with a normal prior.
            covar_module (gpytorch.kernels.Kernel, optional): GP covariance kernel
                class. Defaults to scaled RBF with a gamma prior.
        """
        lb, ub, dim = _process_bounds(lb, ub, dim)
        self.lb, self.ub, self.dim = lb, ub, dim
        self.max_fit_time = max_fit_time
        if likelihood is None:
            likelihood = BernoulliLikelihood()

        inducing_min = lb
        inducing_max = ub
        inducing_points = make_scaled_sobol(inducing_min, inducing_max, inducing_size)

        variational_distribution = MeanFieldVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        )
        super().__init__(variational_strategy)

        mean_prior = inducing_max - inducing_min

        mean_module = mean_module or gpytorch.means.ConstantMean(
            prior=gpytorch.priors.NormalPrior(loc=0.0, scale=2.0)
        )
        ls_prior = gpytorch.priors.GammaPrior(concentration=3.0, rate=6.0 / mean_prior)
        ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate
        ls_constraint = gpytorch.constraints.Positive(
            transform=None, initial_value=ls_prior_mode
        )
        ndim = mean_prior.shape[0]
        covar_module = covar_module or gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint,
                ard_num_dims=ndim,
            ),
            outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
        )

        self.bounds_ = torch.stack([self.lb, self.ub])

        self.mean_module = mean_module
        self.covar_module = covar_module
        self.likelihood = likelihood

    @classmethod
    def from_config(cls, config: Config) -> GPClassificationModel:
        """Alternate constructor for GPClassification model.

        This is used when we recursively build a full sampling strategy
        from a configuration. TODO: document how this works in some tutorial.

        Args:
            config (Config): A configuration containing keys/values matching this class

        Returns:
            GPClassificationModel: Configured class instance.
        """

        classname = cls.__name__
        inducing_size = config.getint(classname, "inducing_size", fallback=10)

        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = config.getint(classname, "dim", fallback=None)

        mean_covar_factory = config.getobj(
            classname, "mean_covar_factory", fallback=default_mean_covar_factory
        )

        mean, covar = mean_covar_factory(config)
        max_fit_time = config.getfloat(classname, "max_fit_time", fallback=None)

        return cls(
            lb=lb,
            ub=ub,
            dim=dim,
            inducing_size=inducing_size,
            mean_module=mean,
            covar_module=covar,
            max_fit_time=max_fit_time,
        )

    def set_train_data(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Set the training data for the model

        Args:
            x (torch.Tensor): training X points
            y ([type]): Training y points
        """
        self.train_inputs = (x,)
        self.train_targets = y

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """Fit underlying model.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.LongTensor): Responses.
        """
        n = train_y.shape[0]
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, n)
        self.train()
        self.set_train_data(train_x, train_y)

        if self.max_fit_time is not None:
            # figure out how long evaluating a single samp
            starttime = time.time()
            _ = mll(self(train_x), train_y)
            single_eval_time = time.time() - starttime
            n_eval = self.max_fit_time // single_eval_time
            options = {"maxfun": n_eval}
            logger.info(f"fit maxfun is {n_eval}")

        else:
            options = {}
        logger.info("Starting fit...")
        starttime = time.time()
        fit_gpytorch_model(mll, options=options)
        logger.info(f"Fit done, time={time.time()-starttime}")

    def sample(
        self, x: Union[torch.Tensor, np.ndarray], num_samples: int
    ) -> torch.Tensor:
        """Sample from underlying model.

        Args:
            x (torch.Tensor): Points at which to sample.
            num_samples (int, optional): Number of samples to return. Defaults to None.
            kwargs are ignored

        Returns:
            torch.Tensor: Posterior samples [num_samples x dim]
        """
        return self.posterior(x).rsample(torch.Size([num_samples])).detach().squeeze()

    def predict(
        self, x: Union[torch.Tensor, np.ndarray], probability_space: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            probability_space (bool, optional): Return outputs in units of
                response probability instead of latent function value. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Posterior mean and variance at queries points.
        """
        if probability_space:
            samps = self.sample(x, num_samples=10000)
            pmean = samps.mean(0).squeeze()
            pvar = samps.var(0).squeeze()
            return pmean, pvar
        else:
            post = self.posterior(x)
            fmean = post.mean.detach().squeeze()
            fvar = post.variance.detach().squeeze()
            return fmean, fvar

    def update(
        self, train_x: torch.Tensor, train_y: torch.Tensor, warmstart: bool = True
    ):
        """Perform a warm-start update of the model from previous fit."""
        self.fit(train_x, train_y)

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
