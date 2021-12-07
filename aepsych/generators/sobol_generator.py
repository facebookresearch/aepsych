#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from aepsych.generators.base import AEPsychGenerator
import torch
import numpy as np
from typing import Union, Optional
from aepsych.utils import make_scaled_sobol, _process_bounds
import warnings
from aepsych.config import Config


class SobolGenerator(AEPsychGenerator):
    """Generator that generates a fixed number of points from the Sobol Sequence."""

    def __init__(
        self,
        n_points: int,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        dim: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Iniatialize SobolGenerator.
        Args:
            lb (Union[np.ndarray, torch.Tensor]): Lower bounds of each parameter.
            lb (Union[np.ndarray, torch.Tensor]): Upper bounds of each parameter.
            n_points (int): Number of points to generate from this generator.
            seed (int, optional): Random seed.
        """

        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)

        if n_points > 0:
            self.points = make_scaled_sobol(
                lb=self.lb, ub=self.ub, size=n_points, seed=seed
            )
        else:
            warnings.warn(
                "SobolStrategy was initialized with n_trials <= 0; it will not generate any points!"
            )
            self.points = np.array([])

        self.n_points = n_points
        self._count = 0
        self.seed = seed

    def gen(self, num_points=1):
        """Query next point(s) to run by quasi-randomly sampling the parameter space.
        Args:
            num_points (int, optional): Number of points to query.
        Returns:
            np.ndarray: Next set of point(s) to evaluate, [num_points x dim].
        """
        if self._count + num_points > self.n_points:
            warnings.warn(
                f"Requesting more points ({num_points}) than"
                + f"this sobol sequence has remaining ({self.n_points-self._count})!"
                + "Giving as many as we have."
            )
            candidates = self.points[self._count :]
        else:
            candidates = self.points[self._count : self._count + num_points]
        self._count = self._count + num_points
        return candidates

    @classmethod
    def from_config(cls, config: Config):
        classname = cls.__name__

        n_points = config.getint(classname, "n_points", fallback=None)
        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = config.getint(classname, "dim", fallback=None)
        seed = config.getint(classname, "seed", fallback=None)

        return cls(n_points=n_points, lb=lb, ub=ub, dim=dim, seed=seed)
