#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gpytorch
import torch
from aepsych.utils import make_scaled_sobol
from aepsych.factory.factory import default_mean_covar_factory
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ApproximateGP
from gpytorch.variational import MeanFieldVariationalDistribution, VariationalStrategy


class GPClassificationModel(ApproximateGP, GPyTorchModel):

    _num_outputs = 1  # let's keep it simple for now

    def __init__(
        self,
        inducing_min,
        inducing_max,
        inducing_size=10,
        mean_module=None,
        covar_module=None,
    ):
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

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

    def set_train_data(self, x, y):
        self.train_inputs = (x,)
        self.train_targets = y

    @classmethod
    def from_config(cls, config):

        classname = cls.__name__
        inducing_min = config.gettensor(classname, "lb")
        inducing_max = config.gettensor(classname, "ub")
        inducing_size = config.getint(classname, "inducing_size", fallback=10)

        mean_covar_factory = config.getobj(classname, "mean_covar_factory", fallback=default_mean_covar_factory)

        mean, covar = mean_covar_factory(config)

        return cls(
            inducing_min=inducing_min,
            inducing_max=inducing_max,
            inducing_size=inducing_size,
            mean_module=mean,
            covar_module=covar,
        )
