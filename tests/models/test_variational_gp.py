#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import numpy.testing as npt
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from sklearn.datasets import make_classification, make_regression

from aepsych.models import BinaryClassificationGP
from aepsych.models.variational_gp import BetaRegressionGP


class BinaryClassificationGPTestCase(unittest.TestCase):
    """
    Super basic smoke test to make sure we know if we broke the underlying model
    for single-probit  ("1AFC") model
    """

    def setUp(self):
        np.random.seed(1)
        torch.manual_seed(1)
        X, y = make_classification(
            n_samples=10,
            n_features=1,
            n_redundant=0,
            n_informative=1,
            random_state=1,
            n_clusters_per_class=1,
        )
        self.X, self.y = torch.Tensor(X), torch.Tensor(y).reshape(-1, 1)

    def test_1d_classification(self):
        """
        Just see if we memorize the training set
        """
        X, y = self.X, self.y
        model = BinaryClassificationGP(
            train_X=X, train_Y=y, likelihood=BernoulliLikelihood(), inducing_points=10
        )
        mll = VariationalELBO(model.likelihood, model.model, len(y))
        fit_gpytorch_mll(mll)

        # pspace
        pm, pv = model.predict(X, probability_space=True)
        pred = (pm > 0.5).numpy()
        npt.assert_allclose(pred.reshape(-1, 1), y)
        npt.assert_array_less(pv, 1)

        # fspace
        pm, pv = model.predict(X, probability_space=False)
        pred = (pm > 0).numpy()
        npt.assert_allclose(pred.reshape(-1, 1), y)
        npt.assert_array_less(1, pv)


class AxBetaRegressionGPTextCase(unittest.TestCase):
    @classmethod
    def setUp(cls):
        np.random.seed(1)
        torch.manual_seed(1)
        X, y = make_regression(
            n_samples=7,
            n_features=3,
            n_informative=1,
            random_state=1,
        )
        # Rescale the target values to the range [0, 1]
        y = (y - y.min()) / (y.max() - y.min())
        cls.X, cls.y = torch.Tensor(X), torch.Tensor(y).reshape(-1, 1)

    def test_1d_regression(self):
        X, y = self.X, self.y
        model = BetaRegressionGP(train_X=X, train_Y=y, inducing_points=10)
        mll = VariationalELBO(model.likelihood, model.model, len(y))
        fit_gpytorch_mll(mll)

        pm, pv = model.predict(X)
        npt.assert_allclose(pm.reshape(-1, 1), y, atol=0.1)
        npt.assert_array_less(pv, 1)


if __name__ == "__main__":
    unittest.main()
