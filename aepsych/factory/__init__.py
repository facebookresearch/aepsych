#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from ..config import Config
from .default import default_mean_covar_factory, DefaultMeanCovarFactory
from .mixed import MixedMeanCovarFactory
from .pairwise import pairwise_mean_covar_factory, PairwiseMeanCovarFactory
from .song import song_mean_covar_factory, SongMeanCovarFactory

"""AEPsych factory functions.
These functions generate a gpytorch Mean and Kernel objects from
aepsych.config.Config configurations, including setting lengthscale
priors and so on. They are primarily used for programmatically
constructing modular AEPsych models from configs.

TODO write a modular AEPsych tutorial.
"""

__all__ = [
    "DefaultMeanCovarFactory",
    "default_mean_covar_factory",
    "MixedMeanCovarFactory",
    "pairwise_mean_covar_factory",
    "PairwiseMeanCovarFactory",
    "SongMeanCovarFactory",
    "song_mean_covar_factory",
]

Config.register_module(sys.modules[__name__])
