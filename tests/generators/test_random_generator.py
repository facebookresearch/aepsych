#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import numpy.testing as npt
from aepsych.config import Config
from aepsych.generators import RandomGenerator


class TestRandomGenerator(unittest.TestCase):
    def test_randomgen_single(self):
        # test that RandomGenerator doesn't mess with shapes
        n = 100
        rand = np.zeros((n, 3))
        mod = RandomGenerator(lb=[1, 2, 3], ub=[2, 3, 4], dim=3)

        for i in range(n):
            rand[i, :] = mod.gen()

        # check that bounds are right
        self.assertTrue(np.all(rand[:, 0] > 1))
        self.assertTrue(np.all(rand[:, 1] > 2))
        self.assertTrue(np.all(rand[:, 2] > 3))
        self.assertTrue(np.all(rand[:, 0] < 2))
        self.assertTrue(np.all(rand[:, 1] < 3))
        self.assertTrue(np.all(rand[:, 2] < 4))

    def test_randomgen_batch(self):
        # test that RandomGenerator doesn't mess with shapes
        n = 100
        mod = RandomGenerator(lb=[1, 2, 3], ub=[2, 3, 4], dim=3)

        rand = mod.gen(n)

        # check that bounds are right
        self.assertTrue((rand[:, 0] > 1).all())
        self.assertTrue((rand[:, 1] > 2).all())
        self.assertTrue((rand[:, 2] > 3).all())
        self.assertTrue((rand[:, 0] < 2).all())
        self.assertTrue((rand[:, 1] < 3).all())
        self.assertTrue((rand[:, 2] < 4).all())

    def test_randomgen_config(self):
        lb = [-1, 0]
        ub = [1, 2]
        config_str = f"""
        [common]
        lb = {lb}
        ub = {ub}
        """
        config = Config(config_str=config_str)
        gen = RandomGenerator.from_config(config)
        npt.assert_equal(gen.lb.numpy(), np.array(lb))
        npt.assert_equal(gen.ub.numpy(), np.array(ub))
        self.assertEqual(gen.dim, len(lb))

    def test_randomgen_fixed(self):
        gen = RandomGenerator(lb=[1, 2, 3], ub=[2, 3, 4], dim=3)
        point = gen.gen(fixed_features={0: 1.5, 1: 2})[0]
        self.assertTrue(point[0] == 1.5)
        self.assertTrue(point[1] == 2)


if __name__ == "__main__":
    unittest.main()
