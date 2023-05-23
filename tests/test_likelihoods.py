#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from aepsych.likelihoods import OrdinalLikelihood
from gpytorch.test.base_likelihood_test_case import BaseLikelihoodTestCase

emtpy_batch_shape = torch.Size([])


class TestOrdinalLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 1
    n_levels = 3

    def _create_targets(self, batch_shape=emtpy_batch_shape):
        res = torch.randint(low=0, high=self.n_levels, size=(*batch_shape, 5)).float()
        return res

    def create_likelihood(self):
        return OrdinalLikelihood(n_levels=self.n_levels)

    def _test_marginal(self, batch_shape=emtpy_batch_shape):
        # disable this test, since Categorical.mean returns nan anyway
        # and we're not overriding this method on base Likelihood
        pass
