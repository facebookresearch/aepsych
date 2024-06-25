#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import numpy.testing as npt
import torch
from aepsych.config import Config
from aepsych.generators import SobolGenerator
from aepsych.utils import make_scaled_sobol


class TestSobolGenerator(unittest.TestCase):
    def test_batchsobol(self):
        mod = SobolGenerator(lb=[1, 2, 3], ub=[2, 3, 4], dim=3, seed=12345)
        acq1 = mod.gen(num_points=2)
        self.assertEqual(acq1.shape, (2, 3))
        acq2 = mod.gen(num_points=3)
        self.assertEqual(acq2.shape, (3, 3))
        acq3 = mod.gen()
        self.assertEqual(acq3.shape, (1, 3))

    def test_sobolgen_single(self):
        # test that SobolGenerator doesn't mess with shapes

        sobol1 = make_scaled_sobol(lb=[1, 2, 3], ub=[2, 3, 4], size=10, seed=12345)

        sobol2 = torch.zeros((10, 3))
        mod = SobolGenerator(lb=[1, 2, 3], ub=[2, 3, 4], dim=3, seed=12345)

        for i in range(10):
            sobol2[i, :] = mod.gen()

        npt.assert_almost_equal(sobol1.numpy(), sobol2.numpy())

        # check that bounds are also right
        self.assertTrue(torch.all(sobol1[:, 0] > 1))
        self.assertTrue(torch.all(sobol1[:, 1] > 2))
        self.assertTrue(torch.all(sobol1[:, 2] > 3))
        self.assertTrue(torch.all(sobol1[:, 0] < 2))
        self.assertTrue(torch.all(sobol1[:, 1] < 3))
        self.assertTrue(torch.all(sobol1[:, 2] < 4))

    def test_sobol_config(self):
        config_str = """
                [common]
                lb = [0]
                ub = [1]
                parnames = [par1]
                stimuli_per_trial = 1

                [SobolGenerator]
                seed=12345
                """
        config = Config()
        config.update(config_str=config_str)
        gen = SobolGenerator.from_config(config)
        npt.assert_equal(gen.lb.numpy(), np.array([0]))
        npt.assert_equal(gen.ub.numpy(), np.array([1]))
        self.assertEqual(gen.seed, 12345)
        self.assertEqual(gen.stimuli_per_trial, 1)

    def test_pairwise_sobol_sizes(self):
        for dim in np.arange(1, 4):
            for nsamp in (3, 5, 7):
                generator = SobolGenerator(
                    lb=np.arange(dim).tolist(),
                    ub=(1 + np.arange(dim)).tolist(),
                    stimuli_per_trial=2,
                )
                shape_out = (nsamp, dim, 2)
                self.assertEqual(generator.gen(nsamp).shape, shape_out)


if __name__ == "__main__":
    unittest.main()
