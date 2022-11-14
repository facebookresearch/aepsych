#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from configparser import NoOptionError
from typing import Tuple

import gpytorch
import torch
from aepsych.config import Config
from aepsych.kernels.rbf_partial_grad import RBFKernelPartialObsGrad
from aepsych.means.constant_partial_grad import ConstantMeanPartialObsGrad
from scipy.stats import norm

"""AEPsych factory functions.
These functions generate a gpytorch Mean and Kernel objects from
aepsych.config.Config configurations, including setting lengthscale
priors and so on. They are primarily used for programmatically
constructing modular AEPsych models from configs.

TODO write a modular AEPsych tutorial.
"""

# AEPsych assumes input dimensions are transformed to [0,1] and we want
# a lengthscale prior that excludes lengthscales that are larger than the
# range of inputs (i.e. >1) or much smaller (i.e. <0.1). This inverse
# gamma prior puts about 99% of the prior probability mass on such values,
# with a preference for small values to prevent oversmoothing. The idea
# is taken from https://betanalpha.github.io/assets/case_studies/gaussian_processes.html#323_Informative_Prior_Model
__default_invgamma_concentration = 4.6
__default_invgamma_rate = 1.0


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
    fixed_mean = config.getboolean(
        "default_mean_covar_factory", "fixed_mean", fallback=False
    )
    lengthscale_prior = config.get(
        "default_mean_covar_factory", "lengthscale_prior", fallback="gamma"
    )
    outputscale_prior = config.get(
        "default_mean_covar_factory", "outputscale_prior", fallback="box"
    )
    kernel = config.getobj(
        "default_mean_covar_factory", "kernel", fallback=gpytorch.kernels.RBFKernel
    )

    assert lb.shape[0] == ub.shape[0], "bounds shape mismatch!"
    dim = lb.shape[0]
    mean = gpytorch.means.ConstantMean()

    if fixed_mean:
        try:
            target = config.getfloat("default_mean_covar_factory", "target")
            mean.constant.requires_grad_(False)
            mean.constant.copy_(torch.tensor(norm.ppf(target)))
        except NoOptionError:
            raise RuntimeError("Config got fixed_mean=True but no target included!")

    if lengthscale_prior == "invgamma":

        ls_prior = gpytorch.priors.GammaPrior(
            concentration=__default_invgamma_concentration,
            rate=__default_invgamma_rate,
            transform=lambda x: 1 / x,
        )

        ls_prior_mode = ls_prior.rate / (ls_prior.concentration + 1)
    elif lengthscale_prior == "gamma":
        ls_prior = gpytorch.priors.GammaPrior(concentration=3.0, rate=6.0)
        ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate
    else:
        raise RuntimeError(
            f"Lengthscale_prior should be invgamma or gamma, got {lengthscale_prior}"
        )

    if outputscale_prior == "gamma":
        os_prior = gpytorch.priors.GammaPrior(concentration=2.0, rate=0.15)
    elif outputscale_prior == "box":
        os_prior = gpytorch.priors.SmoothedBoxPrior(a=1, b=4)
    else:
        raise RuntimeError(
            f"Outputscale_prior should be gamma or box, got {outputscale_prior}"
        )

    ls_constraint = gpytorch.constraints.GreaterThan(
        lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
    )

    covar = gpytorch.kernels.ScaleKernel(
        kernel(
            lengthscale_prior=ls_prior,
            lengthscale_constraint=ls_constraint,
            ard_num_dims=dim,
        ),
        outputscale_prior=os_prior,
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
    assert lb.shape[0] == ub.shape[0], "bounds shape mismatch!"
    dim = lb.shape[0]
    fixed_mean = config.getboolean(
        "monotonic_mean_covar_factory", "fixed_mean", fallback=False
    )

    mean = ConstantMeanPartialObsGrad()

    if fixed_mean:
        try:
            target = config.getfloat("monotonic_mean_covar_factory", "target")
            mean.constant.requires_grad_(False)
            mean.constant.copy_(torch.tensor(norm.ppf(target)))
        except NoOptionError:
            raise RuntimeError("Config got fixed_mean=True but no target included!")

    ls_prior = gpytorch.priors.GammaPrior(
        concentration=__default_invgamma_concentration,
        rate=__default_invgamma_rate,
        transform=lambda x: 1 / x,
    )
    ls_prior_mode = ls_prior.rate / (ls_prior.concentration + 1)
    ls_constraint = gpytorch.constraints.GreaterThan(
        lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
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
    assert lb.shape[0] == ub.shape[0], "bounds shape mismatch!"
    dim = lb.shape[0]

    mean = gpytorch.means.ConstantMean()

    try:
        target = config.getfloat("song_mean_covar_factory", "target")
    except NoOptionError:
        target = 0.75
    mean.constant.requires_grad_(False)
    mean.constant.copy_(torch.tensor(norm.ppf(target)))

    ls_prior = gpytorch.priors.GammaPrior(
        concentration=__default_invgamma_concentration,
        rate=__default_invgamma_rate,
        transform=lambda x: 1 / x,
    )
    ls_prior_mode = ls_prior.rate / (ls_prior.concentration + 1)

    ls_constraint = gpytorch.constraints.GreaterThan(
        lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
    )

    stim_dim = config.getint("song_mean_covar_factory", "stim_dim", fallback=-1)
    context_dims = list(range(dim))
    # if intensity RBF is true, the intensity dimension
    # will have both the RBF and linear kernels
    intensity_RBF = config.getboolean(
        "song_mean_covar_factory", "intensity_RBF", fallback=False
    )
    if not intensity_RBF:
        intensity_dim = 1
        stim_dim = context_dims.pop(stim_dim)  # support relative stim dims
    else:
        intensity_dim = 0
        stim_dim = context_dims[stim_dim]

    # create the LinearKernel
    intensity_covar = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.LinearKernel(active_dims=stim_dim, ard_num_dims=1),
        outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
    )

    if dim == 1:
        # this can just be LinearKernel but for consistency of interface
        # we make it additive with one module
        if not intensity_RBF:
            return (
                mean,
                gpytorch.kernels.AdditiveKernel(intensity_covar),
            )
        else:
            context_covar = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale_prior=ls_prior,
                    lengthscale_constraint=ls_constraint,
                    ard_num_dims=dim,
                    active_dims=context_dims,
                ),
                outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
            )
            return mean, context_covar + intensity_covar
    else:
        context_covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint,
                ard_num_dims=dim - intensity_dim,
                active_dims=context_dims,
            ),
            outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
        )
        return mean, context_covar + intensity_covar


def ordinal_mean_covar_factory(
    config: Config,
) -> Tuple[gpytorch.means.ConstantMean, gpytorch.kernels.ScaleKernel]:

    try:
        base_factory = config.getobj("ordinal_mean_covar_factory", "base_factory")
    except NoOptionError:
        base_factory = default_mean_covar_factory

    _, base_covar = base_factory(config)

    mean = gpytorch.means.ZeroMean()  # wlog since ordinal is shift-invariant

    if isinstance(base_covar, gpytorch.kernels.ScaleKernel):
        covar = base_covar.base_kernel
    else:
        covar = base_covar

    return mean, covar
