#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np
import torch
from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.models.base import AEPsychMixin


class RandomGenerator(AEPsychGenerator):
    """Generator that generates points randomly without an acquisition function."""

    def gen(
        self,
        num_points: int,  # Current implementation only generates 1 point at a time
        model: Optional[AEPsychMixin] = None,
    ) -> np.ndarray:
        """Query next point(s) to run by randomly sampling the parameter space.
        Args:
            num_points (int, optional): Number of points to query.
            model (AEPsychMixin, optional): Included for API compatibility but isn't actually used.
        Returns:
            np.ndarray: Next set of point(s) to evaluate, [num_points x dim].
        """
        X = model.bounds_[0] + torch.rand(model.bounds_.shape[1]) * (
            model.bounds_[1] - model.bounds_[0]
        )
        return X.unsqueeze(0).numpy()

    @classmethod
    def from_config(cls, config: Config):
        return cls()
