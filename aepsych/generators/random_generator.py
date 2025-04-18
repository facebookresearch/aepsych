#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from aepsych.generators.base import AEPsychGenerator
from aepsych.models.base import AEPsychModelMixin
from aepsych.utils import _process_bounds


class RandomGenerator(AEPsychGenerator):
    """Generator that generates points randomly without an acquisition function."""

    _requires_model = False

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        dim: int | None = None,
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
        model: AEPsychModelMixin | None = None,  # included for API compatibility.
        fixed_features: dict[int, float] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Query next point(s) to run by randomly sampling the parameter space.
        Args:
            num_points (int): Number of points to query. Currently, only 1 point can be queried at a time.
            model (AEPsychModelMixin, optional): Model to use for generating points. Not used in this generator.
            fixed_features: (dict[int, float], optional): Parameters that are fixed to specific values.
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
