#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from aepsych.acquisition.rejection_sampler import RejectionSampler
from aepsych.models.derivative_gp import MixedDerivativeVariationalGP
from botorch.utils.testing import BotorchTestCase


class TestRejectionSampling(BotorchTestCase):
    def test_rejection_sampling(self):
        train_X_aug = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
        deriv_constraint_points = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 1.0]]
        )
        train_Y = torch.tensor([[1.0], [2.0], [3.0]])
        m = MixedDerivativeVariationalGP(
            train_x=train_X_aug, train_y=train_Y, inducing_points=train_X_aug
        )
        Xfull = torch.cat((train_X_aug, deriv_constraint_points), dim=0)
        posterior = m.posterior(Xfull)

        sampler = RejectionSampler(
            num_samples=3,
            num_rejection_samples=5000,
            constrained_idx=torch.tensor([3, 4, 5]),
        )
        samples = sampler(posterior)
        self.assertEqual(samples.shape, torch.Size([3, 6, 1]))
        self.assertTrue(torch.all(samples.squeeze(-1)[:, 3:] > 0).item())
