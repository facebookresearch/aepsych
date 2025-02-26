#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from configparser import NoOptionError
from typing import Tuple

import gpytorch
from aepsych.config import Config

from .default import default_mean_covar_factory


def ordinal_mean_covar_factory(
    config: Config,
) -> Tuple[gpytorch.means.ConstantMean, gpytorch.kernels.ScaleKernel]:
    """Create a mean and covariance function for ordinal GPs.

    Args:
        config (Config): Config object containing bounds.

    Returns:
        Tuple[gpytorch.means.ConstantMean, gpytorch.kernels.ScaleKernel]: A tuple containing
        the mean function (ConstantMean) and the covariance function (ScaleKernel).
    """

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
