#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from aepsych.generators.base import AEPsychGenerator
import torch
import numpy as np
from typing import Union, Optional
from aepsych.utils import _process_bounds
from torch.quasirandom import SobolEngine
from aepsych.config import Config
from aepsych.models.base import AEPsychMixin


class SobolGenerator(AEPsychGenerator):
    """Generator that generates points from the Sobol Sequence."""

    _requires_model = False

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
        self.seed = seed
        self.engine = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed)

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
        grid = self.lb + (self.ub - self.lb) * grid
        return grid

    @classmethod
    def from_config(cls, config: Config):
        classname = cls.__name__

        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = config.getint(classname, "dim", fallback=None)
        seed = config.getint(classname, "seed", fallback=None)

        return cls(lb=lb, ub=ub, dim=dim, seed=seed)
