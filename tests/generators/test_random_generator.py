#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from aepsych.generators import RandomGenerator
import numpy as np


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
