#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from ..config import Config
from .epsilon_greedy_generator import EpsilonGreedyGenerator
from .manual_generator import ManualGenerator
from .monotonic_rejection_generator import MonotonicRejectionGenerator
from .monotonic_thompson_sampler_generator import MonotonicThompsonSamplerGenerator
from .optimize_acqf_generator import AxOptimizeAcqfGenerator, OptimizeAcqfGenerator
from .pairwise_optimize_acqf_generator import PairwiseOptimizeAcqfGenerator
from .pairwise_sobol_generator import PairwiseSobolGenerator
from .random_generator import AxRandomGenerator, RandomGenerator
from .semi_p import IntensityAwareSemiPGenerator
from .sobol_generator import AxSobolGenerator, SobolGenerator

__all__ = [
    "OptimizeAcqfGenerator",
    "MonotonicRejectionGenerator",
    "MonotonicThompsonSamplerGenerator",
    "RandomGenerator",
    "SobolGenerator",
    "EpsilonGreedyGenerator",
    "ManualGenerator",
    "PairwiseOptimizeAcqfGenerator",
    "PairwiseSobolGenerator",
    "AxOptimizeAcqfGenerator",
    "AxSobolGenerator",
    "IntensityAwareSemiPGenerator",
    "AxRandomGenerator",
]

Config.register_module(sys.modules[__name__])
