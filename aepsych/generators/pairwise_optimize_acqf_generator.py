#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

from aepsych.config import Config
from aepsych.generators import OptimizeAcqfGenerator


class PairwiseOptimizeAcqfGenerator(OptimizeAcqfGenerator):
    """Deprecated. Use OptimizeAcqfGenerator instead."""

    stimuli_per_trial = 2

    @classmethod
    def from_config(cls, config: Config):
        warnings.warn(
            "PairwiseOptimizeAcqfGenerator is deprecated. Use OptimizeAcqfGenerator instead.",
            DeprecationWarning,
        )
        return super().from_config(config)
