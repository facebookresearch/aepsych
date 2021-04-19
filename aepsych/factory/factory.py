#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from configparser import NoOptionError

import gpytorch
import numpy as np
import torch
from aepsych.kernels.rbf_partial_grad import (
    RBFKernelPartialObsGrad,
)
from aepsych.means.constant_partial_grad import (
    ConstantMeanPartialObsGrad,
)
from scipy.optimize import fsolve
from scipy.stats import invgamma
from scipy.stats import norm


def _compute_invgamma_prior_params_inner(min_lengthscale, max_lengthscale, q=0.01):
    # let's make an invgamma for which 99% of the density lives in our bounds
    def loss(theta):
        alpha, beta = theta
        return (
            invgamma.logcdf(min_lengthscale, alpha, scale=beta) - np.log(q),
            invgamma.logcdf(max_lengthscale, alpha, scale=beta) - np.log(1 - q),
        )

    return fsolve(loss, x0=(1 / max_lengthscale, 6 / max_lengthscale))


def compute_invgamma_prior_params(mins, maxes, q):
    res = [
        _compute_invgamma_prior_params_inner((upper - lower) / 10, upper - lower, q)
        for lower, upper in zip(mins, maxes)
    ]
    alphas = [r[0] for r in res]
    betas = [r[1] for r in res]
    return torch.Tensor(alphas), torch.Tensor(betas)


def default_mean_covar_factory(config):

    lb = config.gettensor("default_mean_covar_factory", "lb")
    ub = config.gettensor("default_mean_covar_factory", "ub")
    mean_prior = ub - lb
    assert lb.shape[0] == ub.shape[0], "bounds shape mismatch!"
    dim = lb.shape[0]
    mean = gpytorch.means.ConstantMean(
        prior=gpytorch.priors.NormalPrior(loc=0, scale=2.0)
    )

    alpha, beta = compute_invgamma_prior_params(mean_prior / 10, mean_prior, q=0.01)

    ls_prior = gpytorch.priors.GammaPrior(
        concentration=alpha, rate=beta, transform=lambda x: 1 / x
    )
    ls_prior_mode = ls_prior.rate / (ls_prior.concentration + 1)
    ls_constraint = gpytorch.constraints.Positive(
        transform=None, initial_value=ls_prior_mode
    )
    covar = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(
            lengthscale_prior=ls_prior,
            lengthscale_constraint=ls_constraint,
            ard_num_dims=dim,
        ),
        outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
    )

    return mean, covar


def monotonic_mean_covar_factory(config):
    lb = config.gettensor("monotonic_mean_covar_factory", "lb")
    ub = config.gettensor("monotonic_mean_covar_factory", "ub")
    mean_prior = ub - lb
    assert lb.shape[0] == ub.shape[0], "bounds shape mismatch!"
    dim = lb.shape[0]

    mean = ConstantMeanPartialObsGrad(
        prior=gpytorch.priors.NormalPrior(loc=0, scale=2.0)
    )

    try:
        target = config.getfloat("monotonic_mean_covar_factory", "target")
    except NoOptionError:
        target = 0.75
    mean.constant.requires_grad_(False)
    mean.constant.copy_(torch.tensor([norm.ppf(target)]))

    alpha, beta = compute_invgamma_prior_params(mean_prior / 10, mean_prior, q=0.01)

    ls_prior = gpytorch.priors.GammaPrior(
        concentration=alpha, rate=beta, transform=lambda x: 1 / x
    )
    ls_prior_mode = ls_prior.rate / (ls_prior.concentration + 1)
    ls_constraint = gpytorch.constraints.Positive(
        transform=None, initial_value=ls_prior_mode
    )

    covar = gpytorch.kernels.ScaleKernel(
        RBFKernelPartialObsGrad(
            lengthscale_prior=ls_prior,
            lengthscale_constraint=ls_constraint,
            ard_num_dims=dim,
        ),
        outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
    )

    return mean, covar


def song_mean_covar_factory(config):
    """
    Factory that makes kernels like Song et al. 2018:
    linear in intensity dimension (assumed to be the last
    dimension), RBF in context dimensions, add them together
    """
    lb = config.gettensor("song_mean_covar_factory", "lb")
    ub = config.gettensor("song_mean_covar_factory", "ub")
    mean_prior = ub - lb
    assert lb.shape[0] == ub.shape[0], "bounds shape mismatch!"
    dim = lb.shape[0]

    mean = gpytorch.means.ConstantMean(
        prior=gpytorch.priors.NormalPrior(loc=0, scale=2.0)
    )
    try:
        target = config.getfloat("song_mean_covar_factory", "target")
    except NoOptionError:
        target = 0.75
    mean.constant.requires_grad_(False)
    mean.constant.copy_(torch.tensor([norm.ppf(target)]))
    # truncate the prior since we only have lengthscales for context dims
    alpha, beta = compute_invgamma_prior_params(
        mean_prior[:-1] / 10, mean_prior[:-1], q=0.01
    )

    ls_prior = gpytorch.priors.GammaPrior(
        concentration=alpha, rate=beta, transform=lambda x: 1 / x
    )
    ls_prior_mode = ls_prior.rate / (ls_prior.concentration + 1)

    ls_constraint = gpytorch.constraints.Positive(
        transform=None, initial_value=ls_prior_mode
    )

    if dim == 1:
        # this can just be LinearKernel but for consistency of interface
        # we make it additive with one module
        return (
            mean,
            gpytorch.kernels.AdditiveKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.LinearKernel(ard_num_dims=1),
                    outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
                )
            ),
        )
    else:
        context_covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint,
                ard_num_dims=dim - 1,
                active_dims=tuple(range(dim - 1)),
            ),
            outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
        )
        intensity_covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel(active_dims=(dim - 1), ard_num_dims=1),
            outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
        )

    return mean, context_covar + intensity_covar
