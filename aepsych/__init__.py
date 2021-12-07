#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from . import (
    acquisition,
    config,
    factory,
    benchmark,
    models,
    strategy,
    utils,
    generators,
)
from .config import Config
from .models import GPClassificationModel
from .strategy import Strategy, SequentialStrategy

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
]

Config.register_module(sys.modules[__name__])
