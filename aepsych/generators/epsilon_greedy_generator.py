#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from aepsych.config import Config

from ..models.base import ModelProtocol
from .base import AEPsychGenerator
from .optimize_acqf_generator import OptimizeAcqfGenerator


class EpsilonGreedyGenerator(AEPsychGenerator):
    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        subgenerator: AEPsychGenerator,
        epsilon: float = 0.1,
    ) -> None:
        """Initialize EpsilonGreedyGenerator.

        Args:
            lb (torch.Tensor): Lower bounds for the optimization.
            ub (torch.Tensor): Upper bounds for the optimization.
            subgenerator (AEPsychGenerator): The generator to use when not exploiting.
            epsilon (float): The probability of exploration. Defaults to 0.1.
        """
        self.subgenerator = subgenerator
        self.epsilon = epsilon
        self.lb = lb
        self.ub = ub

    @classmethod
    def from_config(cls, config: Config) -> "EpsilonGreedyGenerator":
        """Create an EpsilonGreedyGenerator from a Config object.

        Args:
            config (Config): Configuration object containing initialization parameters.

        Returns:
            EpsilonGreedyGenerator: The generator.
        """
        classname = cls.__name__
        lb = torch.tensor(config.getlist(classname, "lb"))
        ub = torch.tensor(config.getlist(classname, "ub"))
        subgen_cls = config.getobj(
            classname, "subgenerator", fallback=OptimizeAcqfGenerator
        )
        subgen = subgen_cls.from_config(config)
        epsilon = config.getfloat(classname, "epsilon", fallback=0.1)
        return cls(lb=lb, ub=ub, subgenerator=subgen, epsilon=epsilon)

    def gen(self, num_points: int, model: ModelProtocol) -> torch.Tensor:
        """Query next point(s) to run by sampling from the subgenerator with probability 1-epsilon, and randomly otherwise.

        Args:
            num_points (int): Number of points to query.
            model (ModelProtocol): Model to use for generating points.
        """
        if num_points > 1:
            raise NotImplementedError("Epsilon-greedy batched gen is not implemented!")
        if np.random.uniform() < self.epsilon:
            sample = np.random.uniform(low=self.lb, high=self.ub)
            return torch.tensor(sample).reshape(1, -1)
        else:
            return self.subgenerator.gen(num_points, model)
