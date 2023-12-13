#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from aepsych.models.exact_gp import ExactGP
from aepsych.models.variational_gp import BinaryClassificationGP
from scipy.special import expit
from scipy.stats import bernoulli


# Fix random seeds
np.random.seed(0)
torch.manual_seed(0)


class SingleOutcomeModelQueryTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bounds = torch.tensor([[0.0], [1.0]])
        x = torch.linspace(0.0, 1.0, 10).reshape(-1, 1)
        y = torch.sin(6.28 * x).reshape(-1, 1)
        cls.model = ExactGP(x, y)
        cls.model.fit()

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

    def test_inverse_query(self):
        bounds = torch.tensor([[0.1], [0.9]])
        val, arg = self.model.inv_query(0.0, bounds)
        # Don't need to be precise since we're working with small data.
        self.assertTrue(-0.01 < val < 0.01)
        self.assertTrue(0.45 < arg < 0.55)

    def test_binary_inverse_query(self):
        X = torch.linspace(-3.0, 3.0, 100).reshape(-1, 1)
        probs = expit(X)
        responses = torch.tensor([float(bernoulli.rvs(p)) for p in probs]).reshape(
            -1, 1
        )

        model = BinaryClassificationGP(X, responses)
        model.fit()

        bounds = torch.tensor([[0.0], [1.0]])
        val, arg = model.inv_query(0.75, bounds, probability_space=True)
        # Don't need to be precise since we're working with small data.
        self.assertTrue(0.7 < val < 0.8)
        self.assertTrue(0 < arg < 2)


class MultiOutcomeModelQueryTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bounds = torch.tensor([[0.0], [1.0]])
        x = torch.linspace(0.0, 1.0, 10).reshape(-1, 1)
        y = torch.cat(
            (
                torch.sin(6.28 * x).reshape(-1, 1),
                torch.cos(6.28 * x).reshape(-1, 1),
            ),
            dim=1,
        )
        cls.model = ExactGP(x, y)
        cls.model.fit()

    def test_max(self):
        mymax, my_argmax = self.model.get_max(self.bounds)
        # Don't need to be precise since we're working with small data.
        self.assertAlmostEqual(mymax.sum().numpy(), np.sqrt(2), places=1)
        self.assertTrue(0.1 < my_argmax < 0.2)

    def test_min(self):
        mymax, my_argmax = self.model.get_min(self.bounds)
        # Don't need to be precise since we're working with small data.
        self.assertAlmostEqual(mymax.sum().numpy(), -np.sqrt(2), places=1)
        self.assertTrue(0.6 < my_argmax < 0.7)

    def test_inverse_query(self):
        bounds = torch.tensor([[0.1], [0.9]])
        val, arg = self.model.inv_query(torch.tensor([0.0, -1]), bounds)
        # Don't need to be precise since we're working with small data.
        self.assertTrue(-0.01 < val[0] < 0.01)
        self.assertTrue(-1.01 < val[1] < -0.99)
        self.assertTrue(0.45 < arg < 0.55)


if __name__ == "__main__":
    unittest.main()
