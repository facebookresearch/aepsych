#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood

from . import (
    acquisition,
    benchmark,
    config,
    factory,
    generators,
    models,
    strategy,
    utils,
)
from .config import Config
from .likelihoods import BernoulliObjectiveLikelihood
from .models import GPClassificationModel
from .strategy import SequentialStrategy, Strategy

__all__ = [
    # modules
    "acquisition",
    "benchmark",
    "config",
    "factory",
    "models",
    "strategy",
    "utils",
    "generators",
    # classes
    "GPClassificationModel",
    "Strategy",
    "SequentialStrategy",
    "BernoulliObjectiveLikelihood",
    "BernoulliLikelihood",
    "GaussianLikelihood",
]

Config.register_module(sys.modules[__name__])
