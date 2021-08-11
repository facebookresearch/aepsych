#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from configparser import NoOptionError

import gpytorch
import numpy as np
import torch
from aepsych.kernels.rbf_partial_grad import RBFKernelPartialObsGrad
from aepsych.means.constant_partial_grad import ConstantMeanPartialObsGrad
from aepsych.config import Config
from scipy.optimize import fsolve
from scipy.stats import invgamma
from scipy.stats import norm
from typing import Tuple, Iterable

"""AEPsych factory functions.
These functions generate a gpytorch Mean and Kernel objects from
aepsych.config.Config configurations, including setting lengthscale
priors and so on. They are primarily used for programmatically
constructing modular AEPsych models from configs.

TODO write a modular AEPsych tutorial.
"""

def _compute_invgamma_prior_params_inner(
    min_lengthscale: float, max_lengthscale: float, q: float = 0.01
) -> Tuple[float, float]:
    """

    Args:
        min_lengthscale (float): Approximate minimum of desired prior.
        max_lengthscale (float): Approximate maximum of desired prior.
        q (float, optional): Proportion of prior to retain outside the
            target range. Defaults to 0.01.

    Returns:
        Tuple[float, float]: alpha and beta params of invgamma distribution.
    """

    def loss(theta):
        alpha, beta = theta
        return (
            invgamma.logcdf(min_lengthscale, alpha, scale=beta) - np.log(q),
            invgamma.logcdf(max_lengthscale, alpha, scale=beta) - np.log(1 - q),
        )

    return fsolve(loss, x0=(1 / max_lengthscale, 6 / max_lengthscale))


def compute_invgamma_prior_params(
    mins: Iterable[float], maxes: Iterable[float], q: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find inverse-gamma prior parameters.

    What we do here is find gamma prior parameters for each element in mins and maxes
    such that most (1-q proportion) of the prior density lives between the target min
    and max.

    Args:
        mins (Iterable[float]): Target minimum lengthscales for each dimension.
        maxes (Iterable[float]): Target maximum lengthscales for each dimension.
        q (float): Proportion of prior that can be outside the target range.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: alpha, beta hyperparam tensors for inverse-gamma prior.
    """
    res = [
        _compute_invgamma_prior_params_inner((upper - lower) / 10, upper - lower, q)
        for lower, upper in zip(mins, maxes)
    ]
    alphas = [r[0] for r in res]
    betas = [r[1] for r in res]
    return torch.Tensor(alphas), torch.Tensor(betas)


def default_mean_covar_factory(
    config: Config,
) -> Tuple[gpytorch.means.ConstantMean, gpytorch.kernels.ScaleKernel]:
    """Default factory for generic GP models

    Args:
        config (Config): Object containing bounds (and potentially other
            config details).

    Returns:
        Tuple[gpytorch.means.Mean, gpytorch.kernels.Kernel]: Instantiated
            ConstantMean and ScaleKernel with priors based on bounds.
    """

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


def monotonic_mean_covar_factory(
    config: Config,
) -> Tuple[ConstantMeanPartialObsGrad, gpytorch.kernels.ScaleKernel]:
    """Default factory for monotonic GP models based on derivative observations.

    Args:
        config (Config): Config containing (at least) bounds, and optionally LSE target.

    Returns:
        Tuple[ConstantMeanPartialObsGrad, gpytorch.kernels.ScaleKernel]: Instantiated mean and
            scaled RBF kernels with partial derivative observations.
    """
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


def song_mean_covar_factory(
    config: Config,
) -> Tuple[gpytorch.means.ConstantMean, gpytorch.kernels.AdditiveKernel]:
    """
    Factory that makes kernels like Song et al. 2018:
    Linear in intensity dimension (assumed to be the last
    dimension), RBF in context dimensions, summed.

    Args:
        config (Config): Config object containing (at least) bounds and optionally
            LSE target.

    Returns:
        Tuple[gpytorch.means.ConstantMean, gpytorch.kernels.AdditiveKernel]: Instantiated
            constant mean object and additive kernel object.
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
