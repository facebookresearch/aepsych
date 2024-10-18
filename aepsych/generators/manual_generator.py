#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.models.base import AEPsychMixin
from aepsych.utils import _process_bounds


class ManualGenerator(AEPsychGenerator):
    """Generator that generates points from a predefined list."""

    _requires_model = False

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        points: Union[np.ndarray, torch.Tensor],
        dim: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        """Iniatialize ManualGenerator.
        Args:
            lb (Union[np.ndarray, torch.Tensor]): Lower bounds of each parameter.
            ub (Union[np.ndarray, torch.Tensor]): Upper bounds of each parameter.
            points (Union[np.ndarray, torch.Tensor]): The points that will be generated.
            dim (int, optional): Dimensionality of the parameter space. If None, it is inferred from lb and ub.
            shuffle (bool): Whether or not to shuffle the order of the points. True by default.
        """
        self.seed = seed
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        if shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(points)
        self.points = torch.tensor(points)
        self.max_asks = len(self.points)
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
        return points

    @classmethod
    def from_config(cls, config: Config, name: Optional[str] = None):
        return cls(**cls.get_config_options(config, name))

    @classmethod
    def get_config_options(cls, config: Config, name: Optional[str] = None) -> Dict:
        if name is None:
            name = cls.__name__

        lb = config.gettensor(name, "lb")
        ub = config.gettensor(name, "ub")
        dim = config.getint(name, "dim", fallback=None)
        points = config.getarray(name, "points")
        shuffle = config.getboolean(name, "shuffle", fallback=True)
        seed = config.getint(name, "seed", fallback=None)

        options = {
            "lb": lb,
            "ub": ub,
            "dim": dim,
            "points": points,
            "shuffle": shuffle,
            "seed": seed,
        }

        return options

    @property
    def finished(self):
        return self._idx >= len(self.points)


class SampleAroundPointsGenerator(ManualGenerator):
    """Generator that samples in a window around reference points in a predefined list."""

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        window: Union[np.ndarray, torch.Tensor],
        points: Union[np.ndarray, torch.Tensor],
        samples_per_point: int,
        dim: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        """Iniatialize SampleAroundPointsGenerator.
        Args:
            lb (Union[np.ndarray, torch.Tensor]): Lower bounds of each parameter.
            ub (Union[np.ndarray, torch.Tensor]): Upper bounds of each parameter.
            window (Union[np.ndarray, torch.Tensor]): How far away to sample from the reference point along each dimension.
            points (Union[np.ndarray, torch.Tensor]): The points that will be generated.
            samples_per_point (int): How many samples around each point to take.
            dim (int, optional): Dimensionality of the parameter space. If None, it is inferred from lb and ub.
            shuffle (bool): Whether or not to shuffle the order of the points. True by default.
            seed (int, optional): Random seed.
        """
        lb, ub, dim = _process_bounds(lb, ub, dim)
        points = torch.Tensor(points)
        self.engine = SobolEngine(dimension=dim, scramble=True, seed=seed)
        generated = []
        for point in points:
            p_lb = torch.max(point - window, lb)
            p_ub = torch.min(point + window, ub)
            grid = self.engine.draw(samples_per_point)
            grid = p_lb + (p_ub - p_lb) * grid
            generated.append(grid)
        generated = torch.Tensor(np.vstack(generated))  # type: ignore

        super().__init__(lb, ub, generated, dim, shuffle, seed)  # type: ignore

    @classmethod
    def get_config_options(cls, config: Config, name: Optional[str] = None) -> Dict:
        if name is None:
            name = cls.__name__

        options = super().get_config_options(config)

        window = config.gettensor(name, "window")
        samples_per_point = config.getint(name, "samples_per_point")

        options.update({"window": window, "samples_per_point": samples_per_point})

        return options
