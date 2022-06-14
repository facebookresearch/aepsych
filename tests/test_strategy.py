#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from aepsych.acquisition.monotonic_rejection import MonotonicMCLSE
from aepsych.generators import MonotonicRejectionGenerator, SobolGenerator
from aepsych.models.gp_classification import GPClassificationModel
from aepsych.models.monotonic_rejection_gp import MonotonicRejectionGP
from aepsych.strategy import SequentialStrategy, Strategy


class TestSequenceGenerators(unittest.TestCase):
    def setUp(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        lb = [-1, -1]
        ub = [1, 1]

        extra_acqf_args = {"target": 0.75, "beta": 1.96}

        self.strat = Strategy(
            model=MonotonicRejectionGP(
                lb=lb,
                ub=ub,
                dim=2,
                monotonic_idxs=[1],
            ),
            generator=MonotonicRejectionGenerator(
                acqf=MonotonicMCLSE, acqf_kwargs=extra_acqf_args
            ),
            min_asks=50,
            lb=lb,
            ub=ub,
            min_post_range=0.3,
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
                min_asks=n, generator=gen, lb=lb, ub=ub, min_total_outcome_occurrences=0
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

        for _ in range(50):
            self.strat.gen()
            self.strat.add_data(np.r_[0.0, 0.0], [0])
        self.assertTrue(self.strat.finished)

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
        lb = [-1, -1]
        ub = [1, 1]

        self.strat = Strategy(
            model=GPClassificationModel(
                lb=lb,
                ub=ub,
            ),
            generator=SobolGenerator(lb=lb, ub=ub),
            min_asks=50,
            lb=lb,
            ub=ub,
        )

        self.strat.keep_most_recent = 2
        data = torch.rand(5, 2)
        for i, d in enumerate(data):
            self.strat.gen()
            self.strat.add_data(d, [0])
            self.strat.fit()

            lb = max(0, i - self.strat.keep_most_recent + 1)
            self.assertTrue(
                torch.equal(self.strat.model.train_inputs[0], data[lb : i + 1])
            )

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
        )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(self.strat.n_trials, 50)


if __name__ == "__main__":
    unittest.main()
