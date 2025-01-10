#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import numpy.testing as npt
from aepsych.config import Config
from aepsych.generators import ManualGenerator, SampleAroundPointsGenerator
from aepsych.transforms import ParameterTransformedGenerator


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
        points = [[10, 10], [10, 11], [11, 10], [11, 11]]
        config_str = f"""
                [common]
                lb = [10, 10]
                ub = [11, 11]
                parnames = [par1, par2]

                [init_strat]
                generator = ManualGenerator

                [ManualGenerator]
                points = {points}
                seed = 123
                """
        config = Config()
        config.update(config_str=config_str)
        gen = ParameterTransformedGenerator.from_config(config, "init_strat")
        npt.assert_equal(gen.lb.numpy(), np.array([0, 0]))
        npt.assert_equal(gen.ub.numpy(), np.array([1, 1]))
        self.assertFalse(gen.finished)

        p1 = list(gen.gen()[0])
        p2 = list(gen.gen()[0])
        p3 = list(gen.gen()[0])
        p4 = list(gen.gen()[0])

        self.assertNotEqual([p1, p2, p3, p4], points)  # make sure it shuffled
        self.assertEqual(sorted([p1, p2, p3, p4]), points)
        self.assertEqual(gen.max_asks, len(points))
        self.assertEqual(gen.seed, 123)
        self.assertTrue(gen.finished)

    def test_manual_generator_fixed(self):
        points = [[10, 10], [10, 11], [11, 10], [11, 11]]
        config_str = f"""
                [common]
                lb = [10, 10]
                ub = [11, 11]
                parnames = [par1, par2]

                [init_strat]
                generator = ManualGenerator

                [ManualGenerator]
                points = {points}
                seed = 123
                """
        config = Config()
        config.update(config_str=config_str)
        gen = ParameterTransformedGenerator.from_config(config, "init_strat")

        with self.assertWarnsRegex(Warning, "Cannot fix features"):
            gen.gen(fixed_features={0: 10.5})


class TestSampleAroundPointsGenerator(unittest.TestCase):
    def test_sample_around_points_generator(self):
        points = [[0.5, 0], [0.5, 1]]
        window = [0.1, 2]
        samples_per_point = 2
        config_str = f"""
                [common]
                lb = [0, 0]
                ub = [1, 1]
                parnames = [par1, par2]

                [SampleAroundPointsGenerator]
                points = {points}
                window = {window}
                samples_per_point = {samples_per_point}
                seed = 123
                """
        config = Config()
        config.update(config_str=config_str)
        gen = SampleAroundPointsGenerator.from_config(config)
        npt.assert_equal(gen.lb.numpy(), np.array([0, 0]))
        npt.assert_equal(gen.ub.numpy(), np.array([1, 1]))
        self.assertEqual(gen.max_asks, len(points * samples_per_point))
        self.assertEqual(gen.seed, 123)
        self.assertFalse(gen.finished)

        points = gen.gen(gen.max_asks)
        for i in range(len(window)):
            npt.assert_array_less(points[:, i], points[:, i] + window[i])
            npt.assert_array_less(np.array([0] * len(points)), points[:, i])
            npt.assert_array_less(points[:, i], np.array([1] * len(points)))

        self.assertTrue(gen.finished)

    def test_sample_around_points_generator_high_dim(self):
        points = [
            [[-1.5, 1], [-1, 1.25], [-2, 1.75]],
            [[-1.25, 1.25], [-1.75, 1.5], [-1.0, 2]],
        ]
        window = [0.25, 0.1]
        samples_per_point = 2
        lb = [-2, 1]
        ub = [-1, 2]
        config_str = f"""
                [common]
                lb = {lb}
                ub = {ub}
                parnames = [par1, par2]
                stimuli_per_trial = 3

                [SampleAroundPointsGenerator]
                points = {points}
                window = {window}
                samples_per_point = {samples_per_point}
                seed = 123
        """
        config = Config()
        config.update(config_str=config_str)
        gen = SampleAroundPointsGenerator.from_config(config)
        npt.assert_equal(gen.lb.numpy(), np.array(lb))
        npt.assert_equal(gen.ub.numpy(), np.array(ub))
        self.assertEqual(gen.max_asks, len(points * samples_per_point))
        self.assertEqual(gen.seed, 123)
        self.assertFalse(gen.finished)

        points = gen.gen(gen.max_asks)
        for i in range(len(window)):
            npt.assert_array_less(points[:, i, :], points[:, i, :] + window[i])
            npt.assert_array_less(
                np.ones(points[:, i, :].shape) * lb[i], points[:, i, :]
            )
            npt.assert_array_less(
                points[:, i, :], np.ones(points[:, i, :].shape) * ub[i]
            )

        self.assertTrue(gen.finished)


if __name__ == "__main__":
    unittest.main()
