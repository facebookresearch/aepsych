#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from aepsych.kernels.rbf_partial_grad import RBFKernelPartialObsGrad
from aepsych.means.constant_partial_grad import ConstantMeanPartialObsGrad
from aepsych.models.derivative_gp import MixedDerivativeVariationalGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.testing import BotorchTestCase
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls.variational_elbo import VariationalELBO


class TestDerivativeGP(BotorchTestCase):
    def testKernel(self):
        K = RBFKernelPartialObsGrad(ard_num_dims=2)
        x1 = torch.cat((torch.rand(5, 2), torch.zeros(5, 1)), dim=1)
        x2 = torch.cat((torch.rand(3, 2), torch.ones(3, 1)), dim=1)
        self.assertEqual(K.forward(x1, x2).shape, torch.Size([5, 3]))

    def testMean(self):
        mu = ConstantMeanPartialObsGrad()
        mu.constant.requires_grad_(False)
        mu.constant.copy_(torch.tensor(5.0))
        mu.constant.requires_grad_(True)

        x1 = torch.cat((torch.rand(5, 2), torch.zeros(5, 1)), dim=1)
        x2 = torch.cat((torch.rand(3, 2), torch.ones(3, 1)), dim=1)
        input = torch.cat((x1, x2))

        z = mu(input)
        self.assertTrue(
            torch.equal(z, torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0]))
        )

    def testMixedDerivativeVariationalGP(self):
        train_x = torch.cat(
            (torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(1), torch.zeros(4, 1)), dim=1
        )
        train_y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        m = MixedDerivativeVariationalGP(
            train_x=train_x,
            train_y=train_y,
            inducing_points=train_x,
            fixed_prior_mean=0.5,
        )
        self.assertEqual(m.mean_module.constant.item(), 0.5)
        self.assertEqual(
            m.covar_module.base_kernel.raw_lengthscale.shape, torch.Size([1, 1])
        )
        mll = VariationalELBO(
            likelihood=BernoulliLikelihood(), model=m, num_data=train_y.numel()
        )
        mll = fit_gpytorch_mll(mll)
        test_x = torch.tensor([[1.0, 0], [3.0, 1.0]])
        m(test_x)
