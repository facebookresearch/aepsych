#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.models.base import AEPsychMixin
from aepsych.utils import _process_bounds


class RandomGenerator(AEPsychGenerator):
    """Generator that generates points randomly without an acquisition function."""

    _requires_model = False

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        dim: Optional[int] = None,
    ) -> None:
        """Iniatialize RandomGenerator.
        Args:
            lb (torch.Tensor): Lower bounds of each parameter.
            ub (torch.Tensor): Upper bounds of each parameter.
            dim (int, optional): Dimensionality of the parameter space. If None, it is inferred from lb and ub.
        """

        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        self.bounds_ = torch.stack([self.lb, self.ub])

    def gen(
        self,
        num_points: int = 1,
        model: Optional[AEPsychMixin] = None,  # included for API compatibility.
        fixed_features: Optional[Dict[int, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Query next point(s) to run by randomly sampling the parameter space.
        Args:
            num_points (int): Number of points to query. Currently, only 1 point can be queried at a time.
            model (AEPsychMixin, optional): Model to use for generating points. Not used in this generator.
            fixed_features: (Dict[int, float], optional): Parameters that are fixed to specific values.
            **kwargs: Ignored, API compatibility

        Returns:
            torch.Tensor: Next set of point(s) to evaluate, [num_points x dim].
        """
        X = self.bounds_[0] + torch.rand((num_points, self.bounds_.shape[1])) * (
            self.bounds_[1] - self.bounds_[0]
        )

        if fixed_features is not None:
            for key, value in fixed_features.items():
                X[:, key] = value

        return X

    @classmethod
    def from_config(cls, config: Config) -> "RandomGenerator":
        """
        Create an instance of RandomGenerator from a configuration object.

        Args:
            config (Config): Configuration object containing initialization parameters.

        Returns:
            RandomGenerator: A configured instance of the generator with specified bounds and dimensionality.
        """
        classname = cls.__name__
        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = config.getint(classname, "dim", fallback=None)
        return cls(lb=lb, ub=ub, dim=dim)
