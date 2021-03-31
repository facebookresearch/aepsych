#!/usr/bin/env python3
import unittest
import numpy as np
import torch
from scipy.stats import norm
from aepsych.acquisition.lse import LevelSetEstimation, MCLevelSetEstimation
from aepsych.acquisition.objective import ProbitObjective

from botorch.utils.testing import MockModel, MockPosterior



class TestLSE(unittest.TestCase):
    def setUp(self):
        f = torch.ones(1) * 1.7
        var = torch.ones(1) * 2.3
        samps = torch.ones(1, 1, 1) * 1.7
        self.model = MockModel(MockPosterior(mean=f, variance=var, samples=samps))

    def test_analytic_lse(self):
        lse = LevelSetEstimation(model=self.model, target=5.0, beta=3.98)
        expected = np.sqrt(3.98) * np.sqrt(2.3) - np.abs(1.7 - 5)
        self.assertAlmostEqual(lse(torch.zeros(1, 1)), expected)

    def test_mclse(self):
        mclse = MCLevelSetEstimation(
            model=self.model, target=5.0, beta=3.98, objective=ProbitObjective()
        )
        expected = np.sqrt(3.98) * np.sqrt(1e-5) - np.abs(norm.cdf(1.7) - 5)
        self.assertAlmostEqual(mclse(torch.zeros(1, 1)), expected)
