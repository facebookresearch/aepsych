#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from aepsych import Config, SequentialStrategy
from aepsych.models.derivative_gp import MixedDerivativeVariationalGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.testing import BotorchTestCase
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls.variational_elbo import VariationalELBO


class TestDerivativeGP(BotorchTestCase):
    def test_MixedDerivativeVariationalGP_gpu(self):
        train_x = torch.cat(
            (torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(1), torch.zeros(4, 1)), dim=1
        )
        train_y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        m = MixedDerivativeVariationalGP(
            train_x=train_x,
            train_y=train_y,
            inducing_points=train_x,
            fixed_prior_mean=0.5,
        ).cuda()

        self.assertEqual(m.mean_module.constant.item(), 0.5)
        self.assertEqual(
            m.covar_module.base_kernel.raw_lengthscale.shape, torch.Size([1, 1])
        )
        mll = VariationalELBO(
            likelihood=BernoulliLikelihood(), model=m, num_data=train_y.numel()
        ).cuda()
        mll = fit_gpytorch_mll(mll)
        test_x = torch.tensor([[1.0, 0], [3.0, 1.0]]).cuda()
        m(test_x)
