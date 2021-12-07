#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from aepsych.utils import make_scaled_sobol
import unittest
from aepsych.generators import SobolGenerator
import torch
import numpy.testing as npt
from aepsych.config import Config


class TestSobolGenerator(unittest.TestCase):
    def test_batchsobol(self):
        mod = SobolGenerator(lb=[1, 2, 3], ub=[2, 3, 4], dim=3, n_points=10, seed=12345)
        acq1 = mod.gen(num_points=2)
        self.assertEqual(acq1.shape, (2, 3))
        acq2 = mod.gen(num_points=3)
        self.assertEqual(acq2.shape, (3, 3))
        acq3 = mod.gen()
        self.assertEqual(acq3.shape, (1, 3))
        with self.assertWarns(Warning):
            mod.gen(num_points=15)

    def test_sobolgen_single(self):
        # test that SobolGenerator doesn't mess with shapes

        sobol1 = make_scaled_sobol(lb=[1, 2, 3], ub=[2, 3, 4], size=10, seed=12345)

        sobol2 = torch.zeros((10, 3))
        mod = SobolGenerator(lb=[1, 2, 3], ub=[2, 3, 4], dim=3, n_points=10, seed=12345)

        npt.assert_equal(sobol1.numpy(), mod.points.numpy())

        for i in range(10):
            sobol2[i, :] = mod.gen()

        npt.assert_equal(sobol1.numpy(), sobol2.numpy())

        # check that bounds are also right
        self.assertTrue(torch.all(sobol1[:, 0] > 1))
        self.assertTrue(torch.all(sobol1[:, 1] > 2))
        self.assertTrue(torch.all(sobol1[:, 2] > 3))
        self.assertTrue(torch.all(sobol1[:, 0] < 2))
        self.assertTrue(torch.all(sobol1[:, 1] < 3))
        self.assertTrue(torch.all(sobol1[:, 2] < 4))

    def test_sobol_config(self):
        for n_trials in [-1, 0, 1]:
            config_str = f"""
                [common]
                lb = [0]
                ub = [1]
                parnames = [par1]

                [SobolGenerator]
                n_points = {n_trials}
                """
            config = Config()
            config.update(config_str=config_str)
            if n_trials <= 0:
                with self.assertWarns(UserWarning):
                    gen = SobolGenerator.from_config(config)
            else:
                gen = SobolGenerator.from_config(config)
            self.assertEqual(gen.n_points, n_trials)
            self.assertEqual(len(gen.points), max(0, n_trials))
