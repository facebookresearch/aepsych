#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from ..config import Config
from .optimize_acqf_generator import OptimizeAcqfGenerator
from .monotonic_rejection_generator import MonotonicRejectionGenerator
from .monotonic_thompson_sampler_generator import MonotonicThompsonSamplerGenerator
from .random_generator import RandomGenerator

__all__ = [
    "OptimizeAcqfGenerator",
    "MonotonicRejectionGenerator",
    "MonotonicThompsonSamplerGenerator",
    "RandomGenerator",
]

Config.register_module(sys.modules[__name__])
