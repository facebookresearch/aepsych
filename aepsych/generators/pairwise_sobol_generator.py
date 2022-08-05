#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Union

import numpy as np
import torch
from aepsych.models.base import AEPsychMixin
from aepsych.utils import _process_bounds
from torch.quasirandom import SobolEngine

from .sobol_generator import SobolGenerator


class PairwiseSobolGenerator(SobolGenerator):
    """Generator that generates pairs of points from the Sobol Sequence."""

    _requires_model = False
    stimuli_per_trial = 2

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        dim: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Iniatialize SobolGenerator.
        Args:
            lb (Union[np.ndarray, torch.Tensor]): Lower bounds of each parameter.
            ub (Union[np.ndarray, torch.Tensor]): Upper bounds of each parameter.
            dim (int, optional): Dimensionality of the parameter space. If None, it is inferred from lb and ub.
            seed (int, optional): Random seed.
        """
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        self._pair_lb = self.lb.repeat(2)
        self._pair_ub = self.ub.repeat(2)
        self.seed = seed
        self.engine = SobolEngine(dimension=self.dim * 2, scramble=True, seed=self.seed)

    def gen(
        self,
        num_points: int = 1,
        model: AEPsychMixin = None,  # included for API compatibility
    ):
        """Query next point(s) to run by quasi-randomly sampling the parameter space.
        Args:
            num_points (int, optional): Number of points to query.
        Returns:
            np.ndarray: Next set of point(s) to evaluate, [num_points x dim].
        """
        grid = self.engine.draw(num_points)
        grid = self._pair_lb + (self._pair_ub - self._pair_lb) * grid
        return torch.tensor(
            np.moveaxis(grid.reshape(num_points, 2, -1).numpy(), -1, -2)
        )
