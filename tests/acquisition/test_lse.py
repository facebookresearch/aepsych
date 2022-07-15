#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from aepsych.acquisition.lse import MCLevelSetEstimation
from aepsych.acquisition.objective import ProbitObjective
from botorch.utils.testing import MockModel, MockPosterior
from scipy.stats import norm


class TestLSE(unittest.TestCase):
    def setUp(self):
        f = torch.ones(1) * 1.7
        var = torch.ones(1) * 2.3
        samps = torch.ones(1, 1, 1) * 1.7
        self.model = MockModel(MockPosterior(mean=f, variance=var, samples=samps))

    def test_mclse(self):
        mclse = MCLevelSetEstimation(
            model=self.model, target=5.0, beta=3.84, objective=ProbitObjective()
        )
        expected = np.sqrt(3.84) * np.sqrt(1e-5) - np.abs(norm.cdf(1.7) - 5)
        self.assertAlmostEqual(mclse(torch.zeros(1, 1)), expected)
