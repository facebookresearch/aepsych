#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from aepsych.config import Config

from ..models.base import ModelProtocol
from .base import AEPsychGenerator
from .optimize_acqf_generator import OptimizeAcqfGenerator


class EpsilonGreedyGenerator(AEPsychGenerator):
    def __init__(self, subgenerator: AEPsychGenerator, epsilon: float = 0.1):
        self.subgenerator = subgenerator
        self.epsilon = epsilon

    @classmethod
    def from_config(cls, config: Config):
        classname = cls.__name__
        subgen_cls = config.getobj(
            classname, "subgenerator", fallback=OptimizeAcqfGenerator
        )
        subgen = subgen_cls.from_config(config)
        epsilon = config.getfloat(classname, "epsilon", fallback=0.1)
        return cls(subgenerator=subgen, epsilon=epsilon)

    def gen(self, num_points: int, model: ModelProtocol):
        if num_points > 1:
            raise NotImplementedError("Epsilon-greedy batched gen is not implemented!")
        if np.random.uniform() < self.epsilon:
            return np.random.uniform(low=model.lb, high=model.ub)
        else:
            return self.subgenerator.gen(num_points, model)
