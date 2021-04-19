#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import torch
from aepsych.modelbridge import MonotonicSingleProbitModelbridge
from aepsych.strategy import (
    SobolStrategy,
    SequentialStrategy,
    ModelWrapperStrategy,
    EpsilonGreedyModelWrapperStrategy,
)
from aepsych.utils import make_scaled_sobol
from aepsych.acquisition.monotonic_rejection import MonotonicMCLSE


class TestSequenceGenerators(unittest.TestCase):
    def test_batchsobol(self):
        mod = SobolStrategy(lb=[1, 2, 3], ub=[2, 3, 4], dim=3, n_trials=10, seed=12345)
        acq1 = mod.gen(num_points=2)
        self.assertEqual(acq1.shape, (2, 3))
        acq2 = mod.gen(num_points=3)
        self.assertEqual(acq2.shape, (3, 3))
        acq3 = mod.gen()
        self.assertEqual(acq3.shape, (1, 3))
        with self.assertWarns(Warning):
            mod.gen(num_points=15)

    def test_sobolmodel_single(self):
        # test that SobolModel doesn't mess with shapes

        sobol1 = make_scaled_sobol(lb=[1, 2, 3], ub=[2, 3, 4], size=10, seed=12345)

        sobol2 = np.zeros((10, 3))
        mod = SobolStrategy(lb=[1, 2, 3], ub=[2, 3, 4], dim=3, n_trials=10, seed=12345)

        npt.assert_equal(sobol1, mod.points)

        for i in range(10):
            sobol2[i, :] = mod.gen()

        npt.assert_equal(sobol1, sobol2)

        # check that bounds are also right
        self.assertTrue(np.all(sobol1[:, 0] > 1))
        self.assertTrue(np.all(sobol1[:, 1] > 2))
        self.assertTrue(np.all(sobol1[:, 2] > 3))
        self.assertTrue(np.all(sobol1[:, 0] < 2))
        self.assertTrue(np.all(sobol1[:, 1] < 3))
        self.assertTrue(np.all(sobol1[:, 2] < 4))

    def test_opt_strategy_single(self):
        strat_list = [
            SobolStrategy(lb=[-1], ub=[1], n_trials=3),
            SobolStrategy(lb=[-10], ub=[-8], n_trials=5),
        ]

        strat = SequentialStrategy(strat_list)
        out = np.zeros(8)
        for i in range(8):
            next_x = strat.gen()
            strat.add_data(next_x, [1])
            out[i] = next_x

        gen1 = out[:3]
        gen2 = out[3:]

        self.assertTrue(np.min(gen1) >= -1)
        self.assertTrue(np.min(gen2) >= -10)
        self.assertTrue(np.max(gen1) <= 1)
        self.assertTrue(np.max(gen2) <= -8)

    def test_warmstart(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        lb = [-1, -1]
        ub = [1, 1]

        extra_acqf_args = {"target": 0.75, "beta": 1.96}

        strat = ModelWrapperStrategy(
            modelbridge=MonotonicSingleProbitModelbridge(
                lb=lb,
                ub=ub,
                dim=2,
                acqf=MonotonicMCLSE,
                extra_acqf_args=extra_acqf_args,
                monotonic_idxs=[1],
            ),
            n_trials=50,
            refit_every=10,
        )
        strat.modelbridge.fit = MagicMock()
        strat.modelbridge.update = MagicMock()
        strat.modelbridge.gen = MagicMock()
        for _ in range(50):
            strat.gen()
            strat.add_data(np.r_[1.0, 1.0], [1])

        assert strat.modelbridge.fit.call_count == 5
        assert strat.modelbridge.update.call_count == 45

    def test_no_warmstart(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        lb = [-1, -1]
        ub = [1, 1]

        extra_acqf_args = {"target": 0.75, "beta": 1.96}
        strat = ModelWrapperStrategy(
            modelbridge=MonotonicSingleProbitModelbridge(
                lb=lb,
                ub=ub,
                dim=2,
                acqf=MonotonicMCLSE,
                extra_acqf_args=extra_acqf_args,
                monotonic_idxs=[1],
            ),
            n_trials=50,
        )
        strat.modelbridge.fit = MagicMock()
        strat.modelbridge.update = MagicMock()
        strat.modelbridge.gen = MagicMock()
        for _ in range(50):
            strat.gen()
            strat.add_data(np.r_[1.0, 1.0], [1])

        assert strat.modelbridge.fit.call_count == 50
        assert strat.modelbridge.update.call_count == 0


class TestEpsilonGreedyStrategy(unittest.TestCase):
    def test_epsilon_greedy(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        lb = [-1, -1]
        ub = [1, 1]
        total_trials = 1000
        extra_acqf_args = {"target": 0.75, "beta": 1.96}
        strat = EpsilonGreedyModelWrapperStrategy(
            modelbridge=MonotonicSingleProbitModelbridge(
                lb=lb,
                ub=ub,
                dim=2,
                acqf=MonotonicMCLSE,
                extra_acqf_args=extra_acqf_args,
                monotonic_idxs=[1],
            ),
            n_trials=total_trials,
        )
        strat.modelbridge.fit = MagicMock()
        strat.modelbridge.update = MagicMock()
        strat.modelbridge.gen = MagicMock()
        for _ in range(total_trials):
            strat.gen()
            strat.add_data(np.r_[1.0, 1.0], [1])

        self.assertTrue(
            np.abs(strat.modelbridge.gen.call_count / total_trials - 0.9) < 0.01
        )

        strat = EpsilonGreedyModelWrapperStrategy(
            modelbridge=MonotonicSingleProbitModelbridge(
                lb=lb,
                ub=ub,
                dim=2,
                acqf=MonotonicMCLSE,
                extra_acqf_args=extra_acqf_args,
                monotonic_idxs=[1],
            ),
            n_trials=total_trials,
            epsilon=0.5,
        )
        strat.modelbridge.fit = MagicMock()
        strat.modelbridge.update = MagicMock()
        strat.modelbridge.gen = MagicMock()
        for _ in range(total_trials):
            strat.gen()
            strat.add_data(np.r_[1.0, 1.0], [1])

        self.assertTrue(
            np.abs(strat.modelbridge.gen.call_count / total_trials - 0.5) < 0.01
        )


if __name__ == "__main__":
    unittest.main()
