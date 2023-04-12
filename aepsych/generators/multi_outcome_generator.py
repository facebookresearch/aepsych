#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Dict

from ax.modelbridge import Models

from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerationStep


class MultiOutcomeOptimizationGenerator(AEPsychGenerationStep):
    @classmethod
    def get_config_options(cls, config: Config, name: str) -> Dict:
        # classname = cls.__name__

        opts = {
            "model": Models.MOO,
        }
        opts.update(super().get_config_options(config, name))

        return opts
