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
    def __init__(self, subgenerator: AEPsychGenerator, epsilon: float = 0.1) -> None:
        """
        Initializes an epsilon-greedy generator with a specified subgenerator and exploration probability.

        Args:
            subgenerator (AEPsychGenerator): The primary generator to produce points outside of epsilon-exploration cases.
            epsilon (float, optional): The probability of exploring by sampling uniformly within the model's bounds.
                Defaults to 0.1.
        """
        self.subgenerator = subgenerator
        self.epsilon = epsilon

    @classmethod
    def from_config(cls, config: Config) -> 'EpsilonGreedyGenerator':
        """
        Creates an instance of EpsilonGreedyGenerator from a configuration object.

        Args:
            config (Config): Configuration object containing initialization parameters.

        Returns:
            EpsilonGreedyGenerator: A configured instance of the generator with specified subgenerator and exploration probability.
        """
        classname = cls.__name__
        subgen_cls = config.getobj(
            classname, "subgenerator", fallback=OptimizeAcqfGenerator
        )
        subgen = subgen_cls.from_config(config)
        epsilon = config.getfloat(classname, "epsilon", fallback=0.1)
        return cls(subgenerator=subgen, epsilon=epsilon)

    def gen(self, num_points: int, model: ModelProtocol) -> torch.Tensor:
        """
        Generates the next query point using an epsilon-greedy strategy.

        Args:
            num_points (int): The number of points to query (must be 1 for this method).
            model (ModelProtocol): The fitted model used to define bounds for exploration.

        Returns:
            torch.Tensor: The next query point, either from the subgenerator or as a random point within model bounds.
        """
        if num_points > 1:
            raise NotImplementedError("Epsilon-greedy batched gen is not implemented!")
        if np.random.uniform() < self.epsilon:
            sample = np.random.uniform(low=model.lb, high=model.ub)
            return torch.tensor(sample).reshape(1, -1)
        else:
            return self.subgenerator.gen(num_points, model)
