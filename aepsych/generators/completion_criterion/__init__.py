#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from aepsych.config import Config

from .min_asks import MinAsks
from .min_total_outcome_occurrences import MinTotalOutcomeOccurrences
from .min_total_tells import MinTotalTells

completion_criteria = [
    MinTotalTells,
    MinAsks,
    MinTotalOutcomeOccurrences,
]

__all__ = [
    "completion_criteria",
    "MinTotalTells",
    "MinAsks",
    "MinTotalOutcomeOccurrences",
]

Config.register_module(sys.modules[__name__])
