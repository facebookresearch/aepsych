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
from aepsych.strategy import Strategy


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

    def test_warmstart(self):
        self.strat.refit_every = 10

        for _ in range(50):
            self.strat.gen()
            self.strat.add_data(np.array([[1.0, 1.0]]), [1])

        self.assertEqual(
            self.strat.model.fit.call_count, 4
        )  # first fit gets skipped because there is no data
        self.assertEqual(self.strat.model.update.call_count, 45)

    def test_no_warmstart(self):
        for _ in range(50):
            self.strat.gen()
            self.strat.add_data(np.array([[1.0, 1.0]]), [1])

        self.assertEqual(
            self.strat.model.fit.call_count, 49
        )  # first fit gets skipped because there is no data
        self.assertEqual(self.strat.model.update.call_count, 0)

    def test_finish_criteria(self):
        for _ in range(49):
            self.strat.gen()
            self.strat.add_data(np.array([[1.0, 1.0]]), [1])
        self.assertFalse(self.strat.finished)

        self.strat.gen()
        self.strat.add_data(np.array([[1.0, 1.0]]), [1])
        self.assertFalse(self.strat.finished)  # not enough "no" trials

        self.strat.gen()
        self.strat.add_data(np.array([[1.0, 1.0]]), [0])
        self.assertFalse(
            self.strat.finished
        )  # not enough difference between posterior min/max

        for _ in range(50):
            self.strat.gen()
            self.strat.add_data(np.array([[1.0, 1.0]]), [0])
        self.assertTrue(self.strat.finished)

    def test_max_asks(self):
        self.strat.max_asks = 50
        for _ in range(49):
            self.strat.gen()
            self.strat.add_data(np.array([[1.0, 1.0]]), [1])
        self.assertFalse(self.strat.finished)

        self.strat.gen()
        self.strat.add_data(np.array([[1.0, 1.0]]), [1])
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
            self.strat.add_data(np.expand_dims(d, axis=0), [0])
            self.strat.fit()

            lb = max(0, i - self.strat.keep_most_recent + 1)
            self.assertTrue(
                torch.equal(self.strat.model.train_inputs[0], data[lb : i + 1])
            )

    def test_add_data(self):
        # this should work
        self.strat.add_data(np.array([[0, 0]]), [0])

        with self.assertRaises(AssertionError):
            # x < lb
            self.strat.add_data(np.array([[-10, 0]]), [1])

        with self.assertRaises(AssertionError):
            # x > ub
            self.strat.add_data(np.array([[0, 10]]), [1])

        with self.assertRaises(AssertionError):
            # y != 0 or 1
            self.strat.add_data(np.array([[-10, 0]]), [-1])


if __name__ == "__main__":
    unittest.main()
