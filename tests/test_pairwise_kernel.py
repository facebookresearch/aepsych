#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import numpy.testing as npt
import torch
from aepsych.kernels.pairwisekernel import PairwiseKernel
from aepsych.kernels.rbf_partial_grad import RBFKernelPartialObsGrad
from gpytorch.kernels import RBFKernel


class PairwiseKernelTest(unittest.TestCase):
    """
    Basic tests that PairwiseKernel is working
    """

    def setUp(self):
        self.latent_kernel = RBFKernel()
        self.kernel = PairwiseKernel(self.latent_kernel)

    def test_kernelgrad_pairwise(self):
        kernel = PairwiseKernel(RBFKernelPartialObsGrad(), is_partial_obs=True)
        x1 = torch.rand(torch.Size([2, 4]))
        x2 = torch.rand(torch.Size([2, 4]))

        x1 = torch.cat((x1, torch.zeros(2, 1)), dim=1)
        x2 = torch.cat((x2, torch.zeros(2, 1)), dim=1)

        deriv_idx_1 = x1[..., -1][:, None]
        deriv_idx_2 = x2[..., -1][:, None]

        a = torch.cat((x1[..., :2], deriv_idx_1), dim=1)
        b = torch.cat((x1[..., 2:-1], deriv_idx_1), dim=1)
        c = torch.cat((x2[..., :2], deriv_idx_2), dim=1)
        d = torch.cat((x2[..., 2:-1], deriv_idx_2), dim=1)

        c12 = kernel.forward(x1, x2).to_dense().detach().numpy()
        pwc = (
            (
                kernel.latent_kernel.forward(a, c)
                - kernel.latent_kernel.forward(a, d)
                - kernel.latent_kernel.forward(b, c)
                + kernel.latent_kernel.forward(b, d)
            )
            .detach()
            .numpy()
        )
        npt.assert_allclose(c12, pwc, atol=1e-6)

    def test_dim_check(self):
        """
        Test that we get expected errors.
        """
        x1 = torch.zeros(torch.Size([3]))
        x2 = torch.zeros(torch.Size([3]))
        x3 = torch.zeros(torch.Size([2]))
        x4 = torch.zeros(torch.Size([4]))

        self.assertRaises(AssertionError, self.kernel.forward, x1=x1, x2=x2)

        self.assertRaises(AssertionError, self.kernel.forward, x1=x3, x2=x4)

    def test_covar(self):
        """
        Test that we get expected covariances
        """
        np.random.seed(1)
        torch.manual_seed(1)

        x1 = torch.rand(torch.Size([2, 4]))
        x2 = torch.rand(torch.Size([2, 4]))
        a = x1[..., :2]
        b = x1[..., 2:]
        c = x2[..., :2]
        d = x2[..., 2:]
        c12 = self.kernel.forward(x1, x2).to_dense().detach().numpy()
        pwc = (
            (
                self.latent_kernel.forward(a, c)
                - self.latent_kernel.forward(a, d)
                - self.latent_kernel.forward(b, c)
                + self.latent_kernel.forward(b, d)
            )
            .detach()
            .numpy()
        )
        npt.assert_allclose(c12, pwc, atol=1e-6)

        shape = np.array(c12.shape)
        npt.assert_equal(shape, np.array([2, 2]))

        x3 = torch.rand(torch.Size([3, 4]))
        x4 = torch.rand(torch.Size([6, 4]))
        a = x3[..., :2]
        b = x3[..., 2:]
        c = x4[..., :2]
        d = x4[..., 2:]
        c34 = self.kernel.forward(x3, x4).to_dense().detach().numpy()
        pwc = (
            (
                self.latent_kernel.forward(a, c)
                - self.latent_kernel.forward(a, d)
                - self.latent_kernel.forward(b, c)
                + self.latent_kernel.forward(b, d)
            )
            .detach()
            .numpy()
        )
        npt.assert_allclose(c34, pwc, atol=1e-6)

        shape = np.array(c34.shape)
        npt.assert_equal(shape, np.array([3, 6]))

    def test_latent_diag(self):
        """
        g(a, a) = 0 for all a, so K((a, a), (a, a)) = 0
        """

        np.random.seed(1)
        torch.manual_seed(1)
        a = torch.rand(torch.Size([2, 2]))

        # should get 0 variance on pairs (a,a)
        diag = torch.cat((a, a), dim=1)
        diagv = self.kernel.forward(diag, diag).to_dense().detach().numpy()
        npt.assert_allclose(diagv, 0.0)

    def test_diag(self):
        """
        make sure the diagonal is the right shape
        """
        np.random.seed(1)
        torch.manual_seed(1)

        x1 = torch.rand(torch.Size([2, 2, 4]))
        x2 = torch.rand(torch.Size([2, 2, 4]))

        diag = self.kernel(x1, x2, diag=True)
        shape = np.array(diag.shape)
        npt.assert_equal(shape, np.array([2, 2]))

        x1 = torch.rand(torch.Size([2, 4]))
        x2 = torch.rand(torch.Size([2, 4]))

        diag = self.kernel(x1, x2, diag=True)
        shape = np.array(diag.shape)
        npt.assert_equal(shape, np.array([2]))


if __name__ == "__main__":
    unittest.main()
