#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from configparser import NoOptionError
from typing import List, Optional, Tuple

import gpytorch
import torch
from aepsych.config import Config
from scipy.stats import norm

from .utils import __default_invgamma_concentration, __default_invgamma_rate

# The gamma lengthscale prior is taken from
# https://betanalpha.github.io/assets/case_studies/gaussian_processes.html#323_Informative_Prior_Model

# The lognormal lengscale prior is taken from
# https://arxiv.org/html/2402.02229v3


def default_mean_covar_factory(
    config: Optional[Config] = None,
    dim: Optional[int] = None,
    stimuli_per_trial: int = 1,
) -> Tuple[gpytorch.means.ConstantMean, gpytorch.kernels.ScaleKernel]:
    """Default factory for generic GP models

    Args:
        config (Config, optional): Object containing bounds (and potentially other
            config details).
        dim (int, optional): Dimensionality of the parameter space. Must be provided
            if config is None.
        stimuli_per_trial (int): Number of stimuli per trial. Defaults to 1.

    Returns:
        Tuple[gpytorch.means.Mean, gpytorch.kernels.Kernel]: Instantiated
            ConstantMean and ScaleKernel with priors based on bounds.
    """

    assert (config is not None) or (
        dim is not None
    ), "Either config or dim must be provided!"

    assert stimuli_per_trial in (1, 2), "stimuli_per_trial must be 1 or 2!"

    zero_mean = False
    if config is not None:
        lb = config.gettensor("default_mean_covar_factory", "lb")
        ub = config.gettensor("default_mean_covar_factory", "ub")
        assert lb.shape[0] == ub.shape[0], "bounds shape mismatch!"
        config_dim: int = lb.shape[0]

        if dim is not None:
            assert dim == config_dim, "Provided config does not match provided dim!"
        else:
            dim = config_dim

        zero_mean = config.getboolean(
            "default_mean_covar_factory", "zero_mean", fallback=False
        )

    mean = _get_default_mean_function(config, zero_mean)
    covar = _get_default_cov_function(config, dim, stimuli_per_trial)  # type: ignore

    return mean, covar


def _get_default_mean_function(
    config: Optional[Config] = None, zero_mean: bool = False
) -> gpytorch.means.ConstantMean:
    """Creates a default mean function for Gaussian Processes.

    Args:
        config (Config, optional): Configuration object.

    Returns:
        gpytorch.means.ConstantMean: An instantiated mean function with appropriate priors based on the configuration.
    """
    if zero_mean:
        mean = gpytorch.means.ZeroMean()
    else:
        mean = gpytorch.means.ConstantMean()

    if config is not None:
        fixed_mean = config.getboolean(
            "default_mean_covar_factory", "fixed_mean", fallback=False
        )
        if fixed_mean:
            if zero_mean:
                warnings.warn(
                    "Specified both `zero_mean = True` and `fixed_mean = True`. Deferring to fixed_mean!"
                )
            try:
                target = config.getfloat("default_mean_covar_factory", "target")
                mean.constant.requires_grad_(False)
                mean.constant.copy_(torch.tensor(norm.ppf(target)))
            except NoOptionError:
                raise RuntimeError("Config got fixed_mean=True but no target included!")

    return mean


def _get_default_cov_function(
    config: Optional[Config],
    dim: int,
    stimuli_per_trial: int,
    active_dims: Optional[List[int]] = None,
) -> gpytorch.kernels.Kernel:
    """Creates a default covariance function for Gaussian Processes.

    Args:
        config (Config, optional): Configuration object.
        dim (int): Dimensionality of the parameter space.
        stimuli_per_trial (int): Number of stimuli per trial.
        active_dims (List[int], optional): List of dimensions to use in the covariance function. Defaults to None.

    Returns:
        gpytorch.kernels.Kernel: An instantiated kernel with appropriate priors based on the configuration.
    """

    # default priors
    lengthscale_prior = "lognormal" if stimuli_per_trial == 1 else "gamma"
    ls_loc = torch.tensor(math.sqrt(2.0), dtype=torch.float64)
    ls_scale = torch.tensor(math.sqrt(3.0), dtype=torch.float64)
    fixed_kernel_amplitude = True if stimuli_per_trial == 1 else False
    outputscale_prior = "box"
    kernel = gpytorch.kernels.RBFKernel

    if config is not None:
        lengthscale_prior = config.get(
            "default_mean_covar_factory",
            "lengthscale_prior",
            fallback=lengthscale_prior,
        )
        if lengthscale_prior == "lognormal":
            ls_loc = config.gettensor(
                "default_mean_covar_factory",
                "ls_loc",
                fallback=ls_loc,
            )
            ls_scale = config.gettensor(
                "default_mean_covar_factory", "ls_scale", fallback=ls_scale
            )
        fixed_kernel_amplitude = config.getboolean(
            "default_mean_covar_factory",
            "fixed_kernel_amplitude",
            fallback=fixed_kernel_amplitude,
        )
        outputscale_prior = config.get(
            "default_mean_covar_factory",
            "outputscale_prior",
            fallback=outputscale_prior,
        )

        kernel = config.getobj("default_mean_covar_factory", "kernel", fallback=kernel)

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

    elif lengthscale_prior == "lognormal":
        if not isinstance(ls_loc, torch.Tensor):
            ls_loc = torch.tensor(ls_loc, dtype=torch.float64)
        if not isinstance(ls_scale, torch.Tensor):
            ls_scale = torch.tensor(ls_scale, dtype=torch.float64)
        ls_prior = gpytorch.priors.LogNormalPrior(ls_loc + math.log(dim) / 2, ls_scale)
        ls_prior_mode = torch.exp(ls_loc - ls_scale**2)
    else:
        raise RuntimeError(
            f"Lengthscale_prior should be invgamma, gamma, or lognormal, got {lengthscale_prior}"
        )

    ls_constraint = gpytorch.constraints.GreaterThan(
        lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
    )

    covar = kernel(
        lengthscale_prior=ls_prior,
        lengthscale_constraint=ls_constraint,
        ard_num_dims=dim,
        active_dims=active_dims,
    )
    if not fixed_kernel_amplitude:
        if outputscale_prior == "gamma":
            os_prior = gpytorch.priors.GammaPrior(concentration=2.0, rate=0.15)
        elif outputscale_prior == "box":
            os_prior = gpytorch.priors.SmoothedBoxPrior(a=1, b=4)
        else:
            raise RuntimeError(
                f"Outputscale_prior should be gamma or box, got {outputscale_prior}"
            )

        covar = gpytorch.kernels.ScaleKernel(
            covar,
            outputscale_prior=os_prior,
            outputscale_constraint=gpytorch.constraints.GreaterThan(1e-4),
        )
    return covar
