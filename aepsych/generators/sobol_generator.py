#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import numpy as np
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
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        dim: Optional[int] = None,
        seed: Optional[int] = None,
        stimuli_per_trial: int = 1,
    ):
        """Iniatialize SobolGenerator.
        Args:
            lb (Union[np.ndarray, torch.Tensor]): Lower bounds of each parameter.
            ub (Union[np.ndarray, torch.Tensor]): Upper bounds of each parameter.
            dim (int, optional): Dimensionality of the parameter space. If None, it is inferred from lb and ub.
            seed (int, optional): Random seed.
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
    ):
        """Query next point(s) to run by quasi-randomly sampling the parameter space.
        Args:
            num_points (int, optional): Number of points to query.
        Returns:
            np.ndarray: Next set of point(s) to evaluate, [num_points x dim].
        """
        grid = self.engine.draw(num_points)
        grid = self.lb + (self.ub - self.lb) * grid
        if self.stimuli_per_trial == 1:
            return grid

        return torch.tensor(
            np.moveaxis(
                grid.reshape(num_points, self.stimuli_per_trial, -1).numpy(),
                -1,
                -self.stimuli_per_trial,
            )
        )

    @classmethod
    def from_config(cls, config: Config):
        classname = cls.__name__

        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = config.getint(classname, "dim", fallback=None)
        seed = config.getint(classname, "seed", fallback=None)
        stimuli_per_trial = config.getint(classname, "stimuli_per_trial")

        return cls(
            lb=lb, ub=ub, dim=dim, seed=seed, stimuli_per_trial=stimuli_per_trial
        )
