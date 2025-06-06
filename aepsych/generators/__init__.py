#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from ..config import Config
from .acqf_grid_search_generator import AcqfGridSearchGenerator
from .acqf_thompson_sampler_generator import AcqfThompsonSamplerGenerator
from .epsilon_greedy_generator import EpsilonGreedyGenerator
from .independent_acqf_generator import IndependentOptimizeAcqfGenerator
from .manual_generator import ManualGenerator, SampleAroundPointsGenerator
from .optimize_acqf_generator import MixedOptimizeAcqfGenerator, OptimizeAcqfGenerator
from .random_generator import RandomGenerator
from .semi_p import IntensityAwareSemiPGenerator
from .sobol_generator import SobolGenerator

__all__ = [
    "OptimizeAcqfGenerator",
    "MixedOptimizeAcqfGenerator",
    "RandomGenerator",
    "SobolGenerator",
    "EpsilonGreedyGenerator",
    "ManualGenerator",
    "SampleAroundPointsGenerator",
    "IndependentOptimizeAcqfGenerator",
    "IntensityAwareSemiPGenerator",
    "AcqfThompsonSamplerGenerator",
    "AcqfGridSearchGenerator",
]

Config.register_module(sys.modules[__name__])
