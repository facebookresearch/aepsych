#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable

import torch
from aepsych.config import Config
from gpytorch.likelihoods import _OneDimensionalLikelihood


class BernoulliObjectiveLikelihood(_OneDimensionalLikelihood):
    """
    Bernoulli likelihood with a flexible link (objective) defined by a callable (which can be a botorch
    objective)
    """

    def __init__(self, objective: Callable):
        super().__init__()
        self.objective = objective

    def forward(self, function_samples, **kwargs):
        output_probs = self.objective(function_samples)
        return torch.distributions.Bernoulli(probs=output_probs)

    @classmethod
    def from_config(cls, config: Config):
        objective_cls = config.getobj(cls.__name__, "objective")
        objective = objective_cls.from_config(config)
        return cls(objective=objective)
