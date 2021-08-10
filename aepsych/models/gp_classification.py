#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypeVar, Union

import gpytorch
import numpy as np
import torch
from aepsych.config import Config
from aepsych.factory.factory import default_mean_covar_factory
from aepsych.utils import make_scaled_sobol
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ApproximateGP
from gpytorch.variational import MeanFieldVariationalDistribution, VariationalStrategy

ModelType = TypeVar("ModelType", bound="GPClassificationModel")


class GPClassificationModel(ApproximateGP, GPyTorchModel):
    """Probit-GP model with variational inference.

    From a conventional ML perspective this is a GP Classification model,
    though in the psychophysics context it can also be thought of as a
    nonlinear generalization of the standard linear model for 1AFC or
    yes/no trials.

    For more on variational inference, see e.g.
    https://docs.gpytorch.ai/en/v1.1.1/examples/04_Variational_and_Approximate_GPs/
    """

    _num_outputs = 1

    def __init__(
        self,
        inducing_min: Union[np.ndarray, torch.Tensor],
        inducing_max: Union[np.ndarray, torch.Tensor],
        inducing_size: int = 10,
        mean_module: gpytorch.means.Mean = None,
        covar_module: gpytorch.kernels.Kernel = None,
    ):
        """Initialize the GP Classification model

        Args:
            inducing_min (Union[np.ndarray, torch.Tensor]): array of lower bounds of inducing points
            inducing_max (Union[np.ndarray, torch.Tensor]): array of upper bounds of inducing points
            inducing_size (int, optional): Number of inducing points. Defaults to 10.
            mean_module (gpytorch.means.Mean, optional): GP mean class. Defaults
                to a constant with a normal prior.
            covar_module (gpytorch.kernels.Kernel, optional): GP covariance kernel
                class. Defaults to scaled RBF with a gamma prior.
        """
        mean_prior = inducing_max - inducing_min

        inducing_points = torch.Tensor(
            make_scaled_sobol(inducing_min, inducing_max, inducing_size)
        )

        variational_distribution = MeanFieldVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = mean_module or gpytorch.means.ConstantMean(
            prior=gpytorch.priors.NormalPrior(loc=0.0, scale=2.0)
        )
        ls_prior = gpytorch.priors.GammaPrior(concentration=3.0, rate=6.0 / mean_prior)
        ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate
        ls_constraint = gpytorch.constraints.Positive(
            transform=None, initial_value=ls_prior_mode
        )
        ndim = mean_prior.shape[0]
        self.covar_module = covar_module or gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint,
                ard_num_dims=ndim,
            ),
            outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
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

    def set_train_data(self, x: torch.Tensor, y: torch.Tensor):
        """Set the training data for the model

        Args:
            x (torch.Tensor): training X points
            y ([type]): Training y points
        """
        self.train_inputs = (x,)
        self.train_targets = y

    @classmethod
    def from_config(cls, config: Config) -> ModelType:
        """Altenate constructor for GPClassification model.

        This is used when we recursively build a full sampling strategy
        from a configuration. TODO: document how this works in some tutorial.

        Args:
            config (Config): A configuration containing keys/values matching this class

        Returns:
            GPClassificationModel: Configured class instance.
        """

        classname = cls.__name__
        inducing_min = config.gettensor(classname, "lb")
        inducing_max = config.gettensor(classname, "ub")
        inducing_size = config.getint(classname, "inducing_size", fallback=10)

        mean_covar_factory = config.getobj(
            classname, "mean_covar_factory", fallback=default_mean_covar_factory
        )

        mean, covar = mean_covar_factory(config)

        return cls(
            inducing_min=inducing_min,
            inducing_max=inducing_max,
            inducing_size=inducing_size,
            mean_module=mean,
            covar_module=covar,
        )
