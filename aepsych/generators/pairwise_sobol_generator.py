#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

from aepsych.config import Config

from .sobol_generator import SobolGenerator


class PairwiseSobolGenerator(SobolGenerator):
    """Deprecated. Use SobolGenerator instead."""

    stimuli_per_trial = 2

    @classmethod
    def from_config(cls, config: Config):
        warnings.warn(
            "PairwiseSobolGenerator is deprecated. Use SobolGenerator instead.",
            DeprecationWarning,
        )
        return super().from_config(config)
