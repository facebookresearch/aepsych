#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from aepsych.models.exact_gp import ExactGP

# Fix random seeds
np.random.seed(0)
torch.manual_seed(0)


class TestModelQuery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bounds = torch.tensor([[0.0], [1.0]])
        x = torch.linspace(0.0, 1.0, 10).reshape(-1, 1)
        y = torch.sin(6.28 * x).reshape(-1, 1)
        cls.model = ExactGP(x, y)
        mll = ExactMarginalLogLikelihood(cls.model.likelihood, cls.model)
        fit_gpytorch_mll(mll)

    def test_min(self):
        mymin, my_argmin = self.model.get_min(self.bounds)
        # Don't need to be precise since we're working with small data.
        self.assertLess(mymin, -0.9)
        self.assertTrue(0.7 < my_argmin < 0.8)

    def test_max(self):
        mymax, my_argmax = self.model.get_max(self.bounds)
        # Don't need to be precise since we're working with small data.
        self.assertGreater(mymax, 0.9)
        self.assertTrue(0.2 < my_argmax < 0.3)


if __name__ == "__main__":
    unittest.main()
