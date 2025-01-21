#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional

import torch
from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator


class OnesGenerator(AEPsychGenerator):
    def __init__(self, dim: int) -> None:
        """A generator that just always gives back 1s"""
        self.dim = dim

    def gen(
        self,
        num_points: int,
        model=None,
        fixed_features: Optional[Dict[int, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Make ones"""
        return torch.ones([num_points, self.dim])

    @classmethod
    def from_config(cls, config: Config) -> "OnesGenerator":
        dim = len(config.getlist("common", "parnames", element_type=str))

        return cls(dim=dim)


Config.register_object(OnesGenerator)
