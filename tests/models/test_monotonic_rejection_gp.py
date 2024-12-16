#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

# run on single threads to keep us from deadlocking weirdly in CI
if "CI" in os.environ or "SANDCASTLE" in os.environ:
    torch.set_num_threads(1)

from aepsych.acquisition.monotonic_rejection import MonotonicMCLSE
from aepsych.acquisition.objective import ProbitObjective
from aepsych.generators import MonotonicRejectionGenerator
from aepsych.models import MonotonicRejectionGP
from aepsych.strategy import Strategy
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.testing import BotorchTestCase
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood
from scipy.stats import norm


class MonotonicRejectionGPLSETest(BotorchTestCase):
    def testRegression(self):
        # Init
        target = 1.5
        model_gen_options = {"num_restarts": 1, "raw_samples": 3, "epochs": 5}
        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([4.0, 4.0])
        inducing_size = 2

        m = MonotonicRejectionGP(
            lb=lb,
            ub=ub,
            likelihood=GaussianLikelihood(),
            fixed_prior_mean=target,
            monotonic_idxs=[1],
            num_induc=inducing_size,
            num_samples=3,
            num_rejection_samples=4,
        )
        strat = Strategy(
            lb=lb,
            ub=ub,
            model=m,
            generator=MonotonicRejectionGenerator(
                MonotonicMCLSE,
                acqf_kwargs={"target": target},
                model_gen_options=model_gen_options,
                lb=lb,
                ub=ub,
            ),
            min_asks=1,
            stimuli_per_trial=1,
            outcome_types=["binary"],
        )
        # Fit
        train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        train_y = torch.tensor([[1.0], [2.0], [3.0]])
        m.fit(train_x=train_x, train_y=train_y)
        self.assertEqual(m.inducing_points.shape, torch.Size([2, 2]))
        self.assertEqual(m.mean_module.constant.item(), 1.5)
        # Predict
        f, var = m.predict(train_x)
        self.assertEqual(f.shape, torch.Size([3]))
        self.assertEqual(var.shape, torch.Size([3]))
        # Gen
        strat.add_data(train_x, train_y)
        Xopt = strat.gen()
        self.assertEqual(Xopt.shape, torch.Size([1, 2]))
        # Acquisition function
        acq = strat.generator._instantiate_acquisition_fn(m)
        self.assertEqual(acq.deriv_constraint_points.shape, torch.Size([2, 3]))
        self.assertTrue(
            torch.equal(acq.deriv_constraint_points[:, -1], 2 * torch.ones(2))
        )
        self.assertEqual(acq.target, 1.5)
        self.assertTrue(isinstance(acq.objective, IdentityMCObjective))

    def testClassification(self):
        # Init
        target = 0.75
        model_gen_options = {"num_restarts": 1, "raw_samples": 3, "epochs": 5}
        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([4.0, 4.0])
        inducing_size = 2

        m = MonotonicRejectionGP(
            lb=lb,
            ub=ub,
            likelihood=BernoulliLikelihood(),
            fixed_prior_mean=target,
            monotonic_idxs=[1],
            num_induc=inducing_size,
            num_samples=3,
            num_rejection_samples=4,
        )
        strat = Strategy(
            lb=lb,
            ub=ub,
            model=m,
            generator=MonotonicRejectionGenerator(
                MonotonicMCLSE,
                acqf_kwargs={"target": target, "objective": ProbitObjective()},
                model_gen_options=model_gen_options,
                lb=lb,
                ub=ub,
            ),
            min_asks=1,
            stimuli_per_trial=1,
            outcome_types=["binary"],
        )
        # Fit
        train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        train_y = torch.tensor([1.0, 1.0, 0.0])
        m.fit(train_x=train_x, train_y=train_y)
        self.assertEqual(m.inducing_points.shape, torch.Size([2, 2]))
        self.assertAlmostEqual(m.mean_module.constant.item(), norm.ppf(0.75))
        # Predict
        f, var = m.predict(train_x)
        self.assertEqual(f.shape, torch.Size([3]))
        self.assertEqual(var.shape, torch.Size([3]))
        # Gen
        strat.add_data(train_x, train_y)
        Xopt = strat.gen()
        self.assertEqual(Xopt.shape, torch.Size([1, 2]))
        # Acquisition function
        acq = strat.generator._instantiate_acquisition_fn(m)
        self.assertEqual(acq.deriv_constraint_points.shape, torch.Size([2, 3]))
        self.assertTrue(
            torch.equal(acq.deriv_constraint_points[:, -1], 2 * torch.ones(2))
        )
        self.assertEqual(acq.target, 0.75)
        self.assertTrue(isinstance(acq.objective, ProbitObjective))
        # Update
        m.update(train_x=train_x[:2, :2], train_y=train_y[:2], warmstart=True)
        self.assertEqual(m.train_inputs[0].shape, torch.Size([2, 3]))
