#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from aepsych.acquisition.monotonic_rejection import MonotonicMCLSE
from aepsych.acquisition.objective import ProbitObjective
from aepsych.models.derivative_gp import MixedDerivativeVariationalGP
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.testing import BotorchTestCase


class TestMonotonicAcq(BotorchTestCase):
    def test_monotonic_acq(self):
        # Init
        train_X_aug = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
        deriv_constraint_points = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 1.0]]
        )
        train_Y = torch.tensor([[1.0], [2.0], [3.0]])

        m = MixedDerivativeVariationalGP(
            train_x=train_X_aug, train_y=train_Y, inducing_points=train_X_aug
        )
        acq = MonotonicMCLSE(
            model=m,
            deriv_constraint_points=deriv_constraint_points,
            num_samples=5,
            num_rejection_samples=8,
            target=1.9,
        )
        self.assertTrue(isinstance(acq.objective, IdentityMCObjective))
        acq = MonotonicMCLSE(
            model=m,
            deriv_constraint_points=deriv_constraint_points,
            num_samples=5,
            num_rejection_samples=8,
            target=1.9,
            objective=ProbitObjective(),
        )
        # forward
        acq(train_X_aug)
        Xfull = torch.cat((train_X_aug, acq.deriv_constraint_points), dim=0)
        posterior = m.posterior(Xfull)
        samples = acq.sampler(posterior)
        self.assertEqual(samples.shape, torch.Size([5, 6, 1]))
