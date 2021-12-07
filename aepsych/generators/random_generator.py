#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
import torch
from aepsych.models.base import AEPsychModel
import numpy as np


class RandomGenerator(AEPsychGenerator):
    def gen(
        self,
        num_points: int,  # Current implementation only generates 1 point at a time
        model: AEPsychModel,
    ) -> np.ndarray:
        X = model.bounds_[0] + torch.rand(model.bounds_.shape[1]) * (
            model.bounds_[1] - model.bounds_[0]
        )
        return X.unsqueeze(0).numpy()

    @classmethod
    def from_config(cls, config: Config):
        return cls()
