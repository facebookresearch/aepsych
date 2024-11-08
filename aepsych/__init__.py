#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood

from . import (
    acquisition,
    config,
    factory,
    generators,
    models,
    strategy,
    transforms,
    utils,
)
from .config import Config
from .likelihoods import BernoulliObjectiveLikelihood
from .models import GPClassificationModel
from .strategy import SequentialStrategy, Strategy

torch.set_default_dtype(torch.float64)

__all__ = [
    # modules
    "acquisition",
    "config",
    "factory",
    "models",
    "strategy",
    "transforms",
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

try:
    from . import benchmark

    __all__ += ["benchmark"]
except ImportError:
    pass

Config.register_module(sys.modules[__name__])
