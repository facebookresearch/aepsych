#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from aepsych.generators import AcqfThompsonSamplerGenerator
from aepsych.models import GPRegressionModel
from aepsych.utils import dim_grid
from botorch.acquisition import qLogNoisyExpectedImprovement


class TestAcqfThompsonSamplerGenerator(unittest.TestCase):
    def test_generation_probabilities(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)

        lb = -3 * torch.ones(2)
        ub = 3 * torch.ones(2)
        X = dim_grid(lb, ub, gridsize=5)
        y = X.sum(dim=-1)

        model = GPRegressionModel(dim=2)
        model.fit(X, y)

        generator = AcqfThompsonSamplerGenerator(
            acqf=qLogNoisyExpectedImprovement,
            lb=lb,
            ub=ub,
        )
        cands = generator.gen(50, model)

        # Fairly weak test. Just trying to ensure that the model is sampling high values with high probability and
        # low values with low probability.
        self.assertGreater(cands.mean(), 0)
