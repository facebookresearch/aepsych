#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import numpy.testing as npt
from aepsych.config import Config
from aepsych.generators import ManualGenerator


class TestManualGenerator(unittest.TestCase):
    def test_batchmanual(self):
        points = np.random.rand(10, 3)
        mod = ManualGenerator(
            lb=[0, 0, 0], ub=[1, 1, 1], dim=3, points=points, shuffle=False
        )

        npt.assert_allclose(points, mod.points)  # make sure they weren't shuffled

        acq1 = mod.gen(num_points=2)
        self.assertEqual(acq1.shape, (2, 3))
        acq2 = mod.gen(num_points=3)
        self.assertEqual(acq2.shape, (3, 3))
        acq3 = mod.gen()
        self.assertEqual(acq3.shape, (1, 3))

        with self.assertWarns(RuntimeWarning):
            acq4 = mod.gen(num_points=10)
        self.assertEqual(acq4.shape, (4, 3))

    def test_manual_generator(self):
        points = [[0, 0], [0, 1], [1, 0], [1, 1]]
        config_str = f"""
                [common]
                lb = [0, 0]
                ub = [1, 1]
                parnames = [par1, par2]

                [ManualGenerator]
                points = {points}
                """
        config = Config()
        config.update(config_str=config_str)
        gen = ManualGenerator.from_config(config)
        npt.assert_equal(gen.lb.numpy(), np.array([0, 0]))
        npt.assert_equal(gen.ub.numpy(), np.array([1, 1]))

        self.assertFalse(gen.finished)

        p1 = list(gen.gen()[0])
        p2 = list(gen.gen()[0])
        p3 = list(gen.gen()[0])
        p4 = list(gen.gen()[0])

        self.assertEqual(sorted([p1, p2, p3, p4]), points)
        self.assertTrue(gen.finished)


if __name__ == "__main__":
    unittest.main()
