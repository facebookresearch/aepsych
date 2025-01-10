#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Dict, Optional

import torch
from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.models.base import AEPsychMixin
from aepsych.utils import _process_bounds
from torch.quasirandom import SobolEngine


class SobolGenerator(AEPsychGenerator):
    """Generator that generates points from the Sobol Sequence."""

    _requires_model = False

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        dim: Optional[int] = None,
        seed: Optional[int] = None,
        stimuli_per_trial: int = 1,
    ) -> None:
        """Iniatialize SobolGenerator.
        Args:
            lb (torch.Tensor): Lower bounds of each parameter.
            ub (torch.Tensor): Upper bounds of each parameter.
            dim (int, optional): Dimensionality of the parameter space. If None, it is inferred from lb and ub.
            seed (int, optional): Random seed.
            stimuli_per_trial (int): Number of stimuli per trial. Defaults to 1.
        """
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        self.lb = self.lb.repeat(stimuli_per_trial)
        self.ub = self.ub.repeat(stimuli_per_trial)
        self.stimuli_per_trial = stimuli_per_trial
        self.seed = seed
        self.engine = SobolEngine(
            dimension=self.dim * stimuli_per_trial, scramble=True, seed=self.seed
        )

    def gen(
        self,
        num_points: int = 1,
        model: Optional[AEPsychMixin] = None,  # included for API compatibility
        fixed_features: Optional[Dict[int, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Query next point(s) to run by quasi-randomly sampling the parameter space.
        Args:
            num_points (int): Number of points to query. Defaults to 1.
            moodel (AEPsychMixin, optional): Model to use for generating points. Not used in this generator. Defaults to None.
            fixed_features: (Dict[int, float], optional): Parameters that are fixed to specific values.
            **kwargs: Ignored, API compatibility
        Returns:
            torch.Tensor: Next set of point(s) to evaluate, [num_points x dim] or [num_points x dim x stimuli_per_trial] if stimuli_per_trial != 1.
        """
        grid = self.engine.draw(num_points)
        grid = self.lb + (self.ub - self.lb) * grid

        if fixed_features is not None:
            for key, value in fixed_features.items():
                grid[:, key] = value

        if self.stimuli_per_trial == 1:
            return grid

        return grid.reshape(num_points, self.stimuli_per_trial, -1).swapaxes(-1, -2)

    @classmethod
    def from_config(cls, config: Config) -> "SobolGenerator":
        """
        Creates an instance of SobolGenerator from a configuration object.

        Args:
            config (Config): Configuration object containing initialization parameters.

        Returns:
            SobolGenerator: A configured instance of the generator with specified bounds, dimensionality, random seed, and stimuli per trial.
        """

        classname = cls.__name__

        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = config.getint(classname, "dim", fallback=None)
        seed = config.getint(classname, "seed", fallback=None)
        stimuli_per_trial = config.getint(classname, "stimuli_per_trial")

        return cls(
            lb=lb, ub=ub, dim=dim, seed=seed, stimuli_per_trial=stimuli_per_trial
        )
