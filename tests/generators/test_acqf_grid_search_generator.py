#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from aepsych.generators import AcqfGridSearchGenerator
from aepsych.models import GPRegressionModel
from aepsych.utils import dim_grid
from botorch.acquisition import qLogNoisyExpectedImprovement


class TestAcqfGridSearchGenerator(unittest.TestCase):
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

        generator = AcqfGridSearchGenerator(
            acqf=qLogNoisyExpectedImprovement,
            lb=lb,
            ub=ub,
        )
        cands = generator.gen(3, model)

        # The top candidates should all be close to the ub
        self.assertEqual(cands.mean().round(), 3)
