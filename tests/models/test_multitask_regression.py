#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import numpy as np
import torch
from aepsych.generators import SobolGenerator
from aepsych.models import IndependentMultitaskGPRModel, MultitaskGPRModel
from parameterized import parameterized

# run on single threads to keep us from deadlocking weirdly in CI
if "CI" in os.environ or "SANDCASTLE" in os.environ:
    torch.set_num_threads(1)

models = [
    (
        MultitaskGPRModel(
            num_outputs=2,
            rank=2,
            dim=1,
        ),
    ),
    (
        IndependentMultitaskGPRModel(
            num_outputs=2,
            dim=1,
        ),
    ),
]


class MultitaskGPRegressionTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)

        generator = SobolGenerator(lb=[-1], ub=[3], dim=1)
        self.x = generator.gen(50)

        f1 = self.x**3 - 4 * self.x**2 + np.random.normal() * 0.01
        f2 = self.x**2 - 7 * self.x + np.random.normal() * 0.01

        self.f = torch.cat((f1, f2), dim=-1)

        self.xtest = generator.gen(10)

        ytrue1 = self.xtest**3 - 4 * self.xtest**2
        ytrue2 = self.xtest**2 - 7 * self.xtest
        self.ytrue = torch.cat((ytrue1, ytrue2), dim=-1)

    @parameterized.expand(models)
    def test_mtgpr_smoke(self, model):
        model.fit(self.x, self.f)
        ypred, _ = model.predict(self.xtest)

        self.assertTrue(np.allclose(self.ytrue, ypred, atol=0.1))  # loose smoke test


if __name__ == "__main__":
    unittest.main()
