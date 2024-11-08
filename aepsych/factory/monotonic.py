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

from .utils import __default_invgamma_concentration, __default_invgamma_rate


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
