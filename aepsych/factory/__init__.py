#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from ..config import Config
from .default import default_mean_covar_factory
from .monotonic import monotonic_mean_covar_factory
from .ordinal import ordinal_mean_covar_factory
from .pairwise import pairwise_mean_covar_factory
from .song import song_mean_covar_factory

"""AEPsych factory functions.
These functions generate a gpytorch Mean and Kernel objects from
aepsych.config.Config configurations, including setting lengthscale
priors and so on. They are primarily used for programmatically
constructing modular AEPsych models from configs.

TODO write a modular AEPsych tutorial.
"""

__all__ = [
    "default_mean_covar_factory",
    "ordinal_mean_covar_factory",
    "monotonic_mean_covar_factory",
    "song_mean_covar_factory",
    "pairwise_mean_covar_factory",
]

Config.register_module(sys.modules[__name__])
