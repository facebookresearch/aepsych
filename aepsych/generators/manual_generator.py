#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Dict, Optional

import torch
from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.models.base import AEPsychMixin
from aepsych.utils import _process_bounds
from torch.quasirandom import SobolEngine


class ManualGenerator(AEPsychGenerator):
    """Generator that generates points from a predefined list."""

    _requires_model = False

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        points: torch.Tensor,
        dim: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """Iniatialize ManualGenerator.
        Args:
            lb (torch.Tensor): Lower bounds of each parameter.
            ub (torch.Tensor): Upper bounds of each parameter.
            points (torch.Tensor): The points that will be generated.
            dim (int, optional): Dimensionality of the parameter space. If None, it is inferred from lb and ub.
            shuffle (bool): Whether or not to shuffle the order of the points. True by default.
            seed (int, optional): Random seed. Defaults to None.
        """
        self.seed = seed
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        self.points = points
        if shuffle:
            if seed is not None:
                torch.manual_seed(seed)
            self.points = points[torch.randperm(len(points))]

        self.max_asks = len(self.points)
        self._idx = 0

    def gen(
        self,
        num_points: int = 1,
        model: Optional[AEPsychMixin] = None,  # included for API compatibility
        fixed_features: Optional[Dict[int, float]] = None,
        **kwargs,  # Ignored
    ) -> torch.Tensor:
        """Query next point(s) to run by quasi-randomly sampling the parameter space.
        Args:
            num_points (int): Number of points to query. Defaults to 1.
            model (AEPsychMixin, optional): Model to use for generating points. Not used in this generator. Defaults to None.
            fixed_features (Dict[int, float], optional): Ignored, kept for consistent
                API.
            **kwargs: Ignored, API compatibility
        Returns:
            torch.Tensor: Next set of point(s) to evaluate, [num_points x dim].
        """
        if num_points > (len(self.points) - self._idx):
            warnings.warn(
                "Asked for more points than are left in the generator! Giving everthing it has!",
                RuntimeWarning,
            )

        if fixed_features is not None:
            warnings.warn(
                f"Cannot fix features when generating from {self.__class__.__name__}"
            )

        points = self.points[self._idx : self._idx + num_points]
        self._idx += num_points
        return points

    @classmethod
    def from_config(
        cls, config: Config, name: Optional[str] = None
    ) -> "ManualGenerator":
        return cls(**cls.get_config_options(config, name))

    @classmethod
    def get_config_options(cls, config: Config, name: Optional[str] = None) -> Dict:
        """
        Extracts and processes configuration options for initializing the ManualGenerator.

        Args:
            config (Config): Configuration object containing initialization parameters.
            name (str, optional): Name of the configuration section for this generator. Defaults to the class name.

        Returns:
            Dict: A dictionary of options, including:
                 "lb" (torch.Tensor): Lower bounds of each parameter.
                 "ub" (torch.Tensor): Upper bounds of each parameter.
                 "dim" (int, optional): Dimensionality of the parameter space.
                 "points" (torch.Tensor): Predefined points to generate.
                 "shuffle" (bool): Whether to shuffle the order of points.
                 "seed" (int, optional): Random seed for shuffling.
        """
        if name is None:
            name = cls.__name__

        lb = config.gettensor(name, "lb")
        ub = config.gettensor(name, "ub")
        dim = config.getint(name, "dim", fallback=None)
        points = config.gettensor(name, "points")
        shuffle = config.getboolean(name, "shuffle", fallback=True)
        seed = config.getint(name, "seed", fallback=None)

        if len(points.shape) == 3:
            # Configs have a reasonable natural input method that produces incorrect tensors
            points = points.swapaxes(-1, -2)

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
    def finished(self) -> bool:
        return self._idx >= len(self.points)


class SampleAroundPointsGenerator(ManualGenerator):
    """Generator that samples in a window around reference points in a predefined list."""

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        window: torch.Tensor,
        points: torch.Tensor,
        samples_per_point: int,
        dim: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """Iniatialize SampleAroundPointsGenerator.
        Args:
            lb (torch.Tensor): Lower bounds of each parameter.
            ub (torch.Tensor): Upper bounds of each parameter.
            window (torch.Tensor): How far away to sample from the reference point along each dimension.
            points (torch.Tensor): The points that will be generated.
            samples_per_point (int): How many samples around each point to take.
            dim (int, optional): Dimensionality of the parameter space. If None, it is inferred from lb and ub.
            shuffle (bool): Whether or not to shuffle the order of the points. True by default.
            seed (int, optional): Random seed.
        """
        lb, ub, dim = _process_bounds(lb, ub, dim)
        self.engine = SobolEngine(dimension=dim, scramble=True, seed=seed)
        gen_points = []
        if len(points.shape) > 2:
            # We need to determine how many stimuli there are per trial to maintain the proper tensor shape
            n_draws = points.shape[-1]
        else:
            n_draws = 1
        for point in points:
            if len(points.shape) > 2:
                point = point.T
            p_lb = torch.max(point - window, lb)
            p_ub = torch.min(point + window, ub)
            for _ in range(samples_per_point):
                grid = self.engine.draw(n_draws)
                grid = p_lb + (p_ub - p_lb) * grid
                gen_points.append(grid)
        if len(points.shape) > 2:
            generated = torch.stack(gen_points)
            generated = generated.swapaxes(-2, -1)
        else:
            generated = torch.vstack(gen_points)

        super().__init__(lb, ub, generated, dim, shuffle, seed)  # type: ignore

    @classmethod
    def get_config_options(cls, config: Config, name: Optional[str] = None) -> Dict:
        """
        Extracts and processes configuration options for initializing the SampleAroundPointsGenerator.

        Args:
            config (Config): Configuration object containing initialization parameters.
            name (str, optional): Name of the configuration section for this generator. Defaults to the class name.

        Returns:
            Dict: A dictionary of options, including:
                - "lb" (torch.Tensor): Lower bounds of each parameter.
                - "ub" (torch.Tensor): Upper bounds of each parameter.
                - "dim" (int, optional): Dimensionality of the parameter space.
                - "points" (torch.Tensor): Predefined reference points.
                - "shuffle" (bool): Whether to shuffle the order of points.
                - "seed" (int, optional): Random seed for shuffling.
                - "window" (torch.Tensor): Sampling range around each reference point along each dimension.
                - "samples_per_point" (int): Number of samples to generate around each reference point.
        """
        if name is None:
            name = cls.__name__

        options = super().get_config_options(config)

        window = config.gettensor(name, "window")
        samples_per_point = config.getint(name, "samples_per_point")

        options.update({"window": window, "samples_per_point": samples_per_point})

        return options
