#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from sklearn.datasets import make_classification
import numpy as np
import torch
import numpy.testing as npt
from aepsych.models import GPClassificationModel


class GPClassificationSmoketest(unittest.TestCase):
    """
    Super basic smoke test to make sure we know if we broke the underlying model
    for single-probit  ("1AFC") model
    """

    def test_1d_classification(self):
        """
        Just see if we memorize the training set
        """
        np.random.seed(1)
        torch.manual_seed(1)
        X, y = make_classification(
            n_features=1,
            n_redundant=0,
            n_informative=1,
            random_state=1,
            n_clusters_per_class=1,
        )
        X, y = torch.Tensor(X), torch.Tensor(y)

        model = GPClassificationModel(torch.Tensor([-3]), torch.Tensor([3]))

        model.fit(X[:50], y[:50])
        pred = (torch.sigmoid(model.posterior(X[:50]).mean) > 0.5).numpy()
        npt.assert_allclose(pred[:, 0], y[:50])

        model.update(X, y)
        pred = (torch.sigmoid(model.posterior(X).mean) > 0.5).numpy()
        npt.assert_allclose(pred[:, 0], y)

if __name__ == "__main__":
    unittest.main()
