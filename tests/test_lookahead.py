#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import product

import numpy as np
import torch
from aepsych.acquisition import (
    ApproxGlobalSUR,
    EAVC,
    GlobalMI,
    GlobalSUR,
    LocalMI,
    LocalSUR,
)
from aepsych.acquisition.bvn import bvn_cdf
from aepsych.acquisition.lookahead_utils import posterior_at_xstar_xq
from botorch.utils.testing import MockModel, MockPosterior
from gpytorch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal


class BvNCDFTestCase(unittest.TestCase):
    def test_bvncdf(self):
        rhos = np.linspace(0.3, 0.9, 7)
        xus = [0.3, 0.5, 0.7]
        yus = [0.3, 0.5, 0.7]

        params = product(rhos, xus, yus)

        for par in params:
            with self.subTest(paraams=params):
                rho, xu, yu = par
                var = np.r_[1, rho, rho, 1].reshape(2, 2)
                x = np.r_[xu, yu]
                scipy_answer = multivariate_normal(cov=var).cdf(x)
                torch_answer = bvn_cdf(
                    torch.tensor(xu), torch.tensor(yu), torch.tensor(rho)
                )
                self.assertTrue(np.isclose(scipy_answer, torch_answer))


class LookaheadPosteriorTestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        np.random.seed(1)
        self.xstar = torch.zeros(1, 1, 1)
        self.xq = torch.randn(1, 2, 1)

        f = torch.rand(3)
        a = torch.rand(3, 3)
        covar = a @ a.T
        flat_diag = torch.rand(3)
        covar = covar + torch.diag_embed(flat_diag)

        mvn = MultivariateNormal(mean=f, covariance_matrix=covar)

        model = MockModel(
            MockPosterior(mean=f[:, None], variance=torch.diag(covar)[None, :, None])
        )
        model._posterior.distribution = mvn
        self.model, self.f, self.covar = model, f, covar

    def test_posterior_extraction(self):
        mu_s, s2_s, mu_q, s2_q, cov_q = posterior_at_xstar_xq(
            self.model, self.xstar, self.xq
        )

        # mean extraction correct
        self.assertTrue(mu_s == self.f[0])
        self.assertTrue((mu_q == self.f[1:]).all())

        # var extraction correct
        self.assertTrue(s2_s == self.covar[0, 0])
        self.assertTrue((s2_q == torch.diag(self.covar)[1:]).all())
        # covar extraction correct
        self.assertTrue((cov_q == self.covar[0, 1:]).all())
        self.assertTrue((cov_q == self.covar[1:, 0]).all())

    def mi_smoketest(self):
        # with the mock posterior, local and global MI should be identical

        local_mi = LocalMI(model=self.model, target=0.75)
        global_mi = GlobalMI(model=self.model, target=0.75, Xq=self.xq[0])
        self.assertTrue(global_mi(self.xstar) == local_mi(self.xstar))

    def sur_smoketest(self):
        # with the mock posterior, local and global SUR should be identical

        local_sur = LocalSUR(model=self.model, target=0.75)
        global_sur = GlobalSUR(model=self.model, target=0.75, Xq=self.xq[0])
        self.assertTrue(global_sur(self.xstar) == local_sur(self.xstar))

    def global_lookahead_smoketest(self):
        for global_lookahead_acq in [
            GlobalMI,
            GlobalSUR,
            ApproxGlobalSUR,
            EAVC,
        ]:
            acq = global_lookahead_acq(model=self.model, target=0.75, Xq=self.xq[0])

            acqval = acq(self.xstar)
            self.assertTrue(acqval.shape == torch.Size([]))
            self.assertTrue(np.isfinite(acqval.numpy()))

    def local_lookahead_smoketest(self):
        for local_lookahead_acq in [
            LocalMI,
            LocalSUR,
        ]:
            acq = local_lookahead_acq(model=self.model, target=0.75)

            acqval = acq(self.xstar)
            self.assertTrue(acqval.shape == torch.Size([]))
            self.assertTrue(np.isfinite(acqval.numpy()))
