#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional, Union

import numpy as np
import torch
from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.models.base import AEPsychMixin
from aepsych.utils import _process_bounds


class ManualGenerator(AEPsychGenerator):
    """Generator that generates points from the Sobol Sequence."""

    _requires_model = False

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        points: Union[np.ndarray, torch.Tensor],
        dim: Optional[int] = None,
        shuffle: bool = True,
    ):
        """Iniatialize SobolGenerator.
        Args:
            lb (Union[np.ndarray, torch.Tensor]): Lower bounds of each parameter.
            ub (Union[np.ndarray, torch.Tensor]): Upper bounds of each parameter.
            points (Union[np.ndarray, torch.Tensor]): The points that will be generated.
            dim (int, optional): Dimensionality of the parameter space. If None, it is inferred from lb and ub.
            shuffle (bool): Whether or not to shuffle the order of the points. True by default.
        """
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        self.points = points
        if shuffle:
            np.random.shuffle(points)
        self.finished = False
        self._idx = 0

    def gen(
        self,
        num_points: int = 1,
        model: Optional[AEPsychMixin] = None,  # included for API compatibility
    ):
        """Query next point(s) to run by quasi-randomly sampling the parameter space.
        Args:
            num_points (int): Number of points to query.
        Returns:
            np.ndarray: Next set of point(s) to evaluate, [num_points x dim].
        """
        if num_points > (len(self.points) - self._idx):
            warnings.warn(
                "Asked for more points than are left in the generator! Giving everthing it has!",
                RuntimeWarning,
            )
        points = self.points[self._idx : self._idx + num_points]
        self._idx += num_points
        if self._idx >= len(self.points):
            self.finished = True
        return points

    @classmethod
    def from_config(cls, config: Config):
        classname = cls.__name__

        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = config.getint(classname, "dim", fallback=None)
        points = config.getarray(classname, "points")
        shuffle = config.getboolean(classname, "shuffle", fallback=True)

        return cls(lb=lb, ub=ub, dim=dim, points=points, shuffle=shuffle)
