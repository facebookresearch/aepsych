#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from ...config import Config
from .fixed import FixedAllocator
from .greedy_variance_reduction import GreedyVarianceReduction
from .kmeans import KMeansAllocator
from .sobol import SobolAllocator

__all__ = [
    "FixedAllocator",
    "GreedyVarianceReduction",
    "KMeansAllocator",
    "SobolAllocator",
]

Config.register_module(sys.modules[__name__])
