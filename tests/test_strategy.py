#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from aepsych.acquisition import MCLevelSetEstimation
from aepsych.acquisition.monotonic_rejection import MonotonicMCLSE
from aepsych.generators import (
    MonotonicRejectionGenerator,
    OptimizeAcqfGenerator,
    SobolGenerator,
)
from aepsych.models.gp_classification import GPClassificationModel
from aepsych.models.monotonic_rejection_gp import MonotonicRejectionGP
from aepsych.strategy import SequentialStrategy, Strategy
from aepsych.transforms import (
    ParameterTransformedGenerator,
    ParameterTransformedModel,
    ParameterTransforms,
)
from aepsych.transforms.ops import NormalizeScale


class TestSequenceGenerators(unittest.TestCase):
    def setUp(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        lb = torch.tensor([-1, -1])
        ub = torch.tensor([1, 1])

        extra_acqf_args = {"target": 0.75, "beta": 1.96}

        transforms = ParameterTransforms(
            normalize=NormalizeScale(d=2, bounds=torch.stack([lb, ub]))
        )

        self.strat = Strategy(
            model=ParameterTransformedModel(
                MonotonicRejectionGP,
                lb=lb,
                ub=ub,
                transforms=transforms,
                monotonic_idxs=[1],
            ),
            generator=ParameterTransformedGenerator(
                MonotonicRejectionGenerator,
                transforms=transforms,
                acqf=MonotonicMCLSE,
                acqf_kwargs=extra_acqf_args,
                lb=lb,
                ub=ub,
            ),
            min_asks=50,
            lb=lb,
            ub=ub,
            min_post_range=0.3,
            stimuli_per_trial=1,
            outcome_types=["binary"],
        )
        self.strat.model.fit = MagicMock()
        self.strat.model.update = MagicMock()
        self.strat.generator.gen = MagicMock()

    def test_opt_strategy_single(self):
        lbs = [[-1], [-10]]
        ubs = [[1], [-8]]
        n = [3, 5]
        strat_list = []
        for lb, ub, n in zip(lbs, ubs, n):
            gen = SobolGenerator(lb, ub)
            strat = Strategy(
                min_asks=n,
                generator=gen,
                lb=lb,
                ub=ub,
                min_total_outcome_occurrences=0,
                stimuli_per_trial=1,
                outcome_types=["binary"],
            )
            strat_list.append(strat)

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
        self.strat.refit_every = 10

        for _ in range(50):
            self.strat.gen()
            self.strat.add_data(np.r_[1.0, 1.0], [1])

        self.assertEqual(
            self.strat.model.fit.call_count, 4
        )  # first fit gets skipped because there is no data
        self.assertEqual(self.strat.model.update.call_count, 45)

    def test_no_warmstart(self):
        for _ in range(50):
            self.strat.gen()
            self.strat.add_data(np.r_[1.0, 1.0], [1])

        self.assertEqual(
            self.strat.model.fit.call_count, 49
        )  # first fit gets skipped because there is no data
        self.assertEqual(self.strat.model.update.call_count, 0)

    def test_finish_criteria(self):
        for _ in range(49):
            self.strat.gen()
            self.strat.add_data(np.r_[1.0, 1.0], [1])
        self.assertFalse(self.strat.finished)
        self.strat.gen()
        self.strat.add_data(np.r_[1.0, 1.0], [1])
        self.assertFalse(self.strat.finished)  # not enough "no" trials

        self.strat.gen()
        self.strat.add_data(np.r_[1.0, 1.0], [0])
        self.assertFalse(
            self.strat.finished
        )  # not enough difference between posterior min/max

    def test_min_post_range(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        lb = [0]
        ub = [1]

        self.strat = Strategy(
            model=GPClassificationModel(
                dim=1,
            ),
            generator=SobolGenerator(lb=lb, ub=ub),
            min_asks=10,
            lb=lb,
            ub=ub,
            stimuli_per_trial=1,
            outcome_types=["binary"],
            min_post_range=0.3,
        )

        loops = 0
        while not self.strat.finished:
            points = self.strat.gen(1)
            response = int(np.random.rand() < points)
            self.strat.add_data(points, torch.tensor([response]))
            loops += 1

            if loops > 50:
                self.fail("min_post_range didn't finish even after 50 loops.")
                break

    def test_max_asks(self):
        self.strat.max_asks = 50
        for _ in range(49):
            self.strat.gen()
            self.strat.add_data(np.r_[1.0, 1.0], [1])
        self.assertFalse(self.strat.finished)

        self.strat.gen()
        self.strat.add_data(np.r_[1.0, 1.0], [1])
        self.assertTrue(self.strat.finished)

    def test_keep_most_recent(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        lb = [-1.0, -1.0]
        ub = [1.0, 1.0]

        self.strat = Strategy(
            model=GPClassificationModel(
                dim=2,
            ),
            generator=SobolGenerator(lb=lb, ub=ub),
            min_asks=50,
            lb=lb,
            ub=ub,
            stimuli_per_trial=1,
            outcome_types=["binary"],
        )

        self.strat.keep_most_recent = 2
        data = torch.rand(4, 2)
        for i, d in enumerate(data):
            self.strat.add_data(d, [0])
            self.strat.update()

            lb = max(0, i - self.strat.keep_most_recent + 1)
            self.assertTrue(
                torch.equal(self.strat.model.train_inputs[0], data[lb : i + 1])
            )

    def test_run_indefinitely(self):
        lb = torch.tensor([-1.0, -1.0])
        ub = torch.tensor([1.0, 1.0])

        with self.assertWarns(UserWarning):
            self.strat = Strategy(
                model=GPClassificationModel(
                    dim=2,
                ),
                generator=SobolGenerator(lb=lb, ub=ub),
                lb=lb,
                ub=ub,
                stimuli_per_trial=1,
                outcome_types=["binary"],
                min_asks=1,  # should be ignored
                run_indefinitely=True,
            )
        self.strat.gen()
        self.assertFalse(self.strat.finished)
        self.strat.finish()
        self.assertTrue(self.strat.finished)

    def test_n_trials_deprecation(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        lb = [-1, -1]
        ub = [1, 1]

        self.strat = Strategy(
            generator=SobolGenerator(lb=lb, ub=ub),
            min_asks=50,
            lb=lb,
            ub=ub,
            stimuli_per_trial=1,
            outcome_types=["binary"],
        )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(self.strat.n_trials, 50)

    def test_batchsobol_pairwise(self):
        lb = [1, 2, 3]
        ub = [2, 3, 4]
        min_asks = 10
        mod = Strategy(
            lb=lb,
            ub=ub,
            generator=SobolGenerator(lb=lb, ub=ub, seed=12345, stimuli_per_trial=2),
            min_asks=min_asks,
            stimuli_per_trial=2,
            outcome_types=["binary"],
        )
        acq1 = mod.gen(num_points=2)
        self.assertEqual(acq1.shape, (2, 3, 2))
        acq2 = mod.gen(num_points=3)
        self.assertEqual(acq2.shape, (3, 3, 2))
        acq3 = mod.gen()
        self.assertEqual(acq3.shape, (1, 3, 2))

    def test_opt_strategy_pairwise(self):
        strat_list = [
            Strategy(
                lb=[-1],
                ub=[1],
                min_asks=3,
                generator=SobolGenerator(lb=[-1], ub=[1], stimuli_per_trial=2),
                stimuli_per_trial=2,
                outcome_types=["binary"],
                min_total_outcome_occurrences=0,
            ),
            Strategy(
                lb=[-10],
                ub=[-8],
                min_asks=5,
                generator=SobolGenerator(lb=[-10], ub=[-8], stimuli_per_trial=2),
                stimuli_per_trial=2,
                outcome_types=["binary"],
                min_total_outcome_occurrences=0,
            ),
        ]

        strat = SequentialStrategy(strat_list)
        out = np.zeros((8, 2))
        for i in range(8):
            next_x = strat.gen()
            strat.add_data(next_x, [1])
            out[i] = next_x

        gen1 = out[:3]
        gen2 = out[3:]

        self.assertTrue(np.min(gen2) >= -10)
        self.assertTrue(np.min(gen1) >= -1)
        self.assertTrue(np.max(gen1) <= 1)
        self.assertTrue(np.max(gen2) <= -8)

    def test_strategy_asserts(self):
        class MockModel(object):
            _num_outputs = 1
            _batch_shape = 2
            stimuli_per_trial = 1
            outcome_type = "binary"

        # assert if model and strategy disagree on stimuli_per_trial
        with self.assertRaises(AssertionError):
            _ = Strategy(
                lb=[-1],
                ub=[1],
                min_asks=5,
                stimuli_per_trial=2,
                model=MockModel(),
                generator=SobolGenerator(lb=[-1], ub=[1], stimuli_per_trial=2),
                outcome_types=["binary"],
            )

        # assert if model and strategy disagree on outcome_type
        with self.assertRaises(AssertionError):
            _ = Strategy(
                lb=[-1],
                ub=[1],
                min_asks=5,
                stimuli_per_trial=1,
                model=MockModel(),
                generator=SobolGenerator(lb=[-1], ub=[1], stimuli_per_trial=1),
                outcome_types=["notbinary"],
            )

        # assert if model and strategy disagree on num outcomes
        with self.assertRaises(AssertionError):
            _ = Strategy(
                lb=[-1],
                ub=[1],
                min_asks=5,
                stimuli_per_trial=1,
                model=MockModel(),
                generator=SobolGenerator(lb=[-1], ub=[1], stimuli_per_trial=1),
                outcome_types=["binary", "extra"],
            )

        try:
            # no assert on 1 stim per trial
            _ = Strategy(
                lb=[-1],
                ub=[1],
                min_asks=5,
                stimuli_per_trial=1,
                model=MockModel(),
                generator=SobolGenerator(lb=[-1], ub=[1], stimuli_per_trial=1),
                outcome_types=["binary"],
            )
            # no assert on 2 stim per trial
            model = MockModel()
            model._num_outputs = 2
            model.outcome_type = ["binary", "extra"]
            _ = Strategy(
                lb=[-1],
                ub=[1],
                min_asks=5,
                stimuli_per_trial=1,
                model=model,
                generator=SobolGenerator(lb=[-1], ub=[1], stimuli_per_trial=2),
                outcome_types=["binary", "extra"],
            )
        except AssertionError:
            self.fail("Strategy raised unexpected AssertionError on __init__!")


class TestStrategyGPU(unittest.TestCase):
    @unittest.skipIf(
        torch.cuda.is_available(), "gpu is available, can't test no gpu warning"
    )
    def test_no_gpu_model_warn(self):
        with self.assertWarns(UserWarning):
            Strategy(
                lb=[0],
                ub=[1],
                stimuli_per_trial=1,
                outcome_types=["binary"],
                model=GPClassificationModel(
                    dim=1,
                ),
                generator=SobolGenerator(lb=[0], ub=[1]),
                use_gpu_modeling=True,
            )

    @unittest.skipIf(
        torch.cuda.is_available(), "gpu is available, can't test no gpu warning"
    )
    def test_no_gpu_generator_warn(self):
        with self.assertWarns(UserWarning):
            Strategy(
                lb=[0],
                ub=[1],
                stimuli_per_trial=1,
                outcome_types=["binary"],
                model=GPClassificationModel(
                    dim=1,
                ),
                generator=OptimizeAcqfGenerator(
                    lb=[0], ub=[1], acqf=MCLevelSetEstimation
                ),
                use_gpu_generating=True,
            )


if __name__ == "__main__":
    unittest.main()
