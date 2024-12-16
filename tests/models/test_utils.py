#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from aepsych.models import GPClassificationModel
from aepsych.models.inducing_points import (
    AutoAllocator,
    FixedAllocator,
    GreedyVarianceReduction,
    KMeansAllocator,
    SobolAllocator,
)
from aepsych.models.utils import select_inducing_points
from sklearn.datasets import make_classification


class UtilsTestCase(unittest.TestCase):
    def test_select_inducing_points(self):
        """Verify that when we have n_induc > data size, we use data as inducing,
        and otherwise we correctly select inducing points."""
        X, y = make_classification(
            n_samples=100,
            n_features=1,
            n_redundant=0,
            n_informative=1,
            random_state=1,
            n_clusters_per_class=1,
        )
        X, y = torch.Tensor(X), torch.Tensor(y)
        inducing_size = 20
        lb = torch.Tensor([-3])
        ub = torch.Tensor([3])
        bounds = torch.stack([lb, ub])

        model = GPClassificationModel(
            dim=1,
            inducing_size=inducing_size,
            inducing_point_method=AutoAllocator(dim=1),
        )
        model.set_train_data(X[:10, ...], y[:10])

        # (inducing point selection sorts the inputs so we sort X to verify)
        self.assertTrue(
            np.allclose(
                select_inducing_points(
                    allocator=AutoAllocator(dim=1),
                    inducing_size=inducing_size,
                    covar_module=model.covar_module,
                    X=model.train_inputs[0],
                ),
                X[:10].sort(0).values,
            )
        )

        model.set_train_data(X, y)

        self.assertTrue(
            len(
                select_inducing_points(
                    allocator=AutoAllocator(dim=1),
                    inducing_size=inducing_size,
                    covar_module=model.covar_module,
                    X=model.train_inputs[0],
                )
            )
            <= 20
        )

        self.assertTrue(
            len(
                select_inducing_points(
                    allocator=GreedyVarianceReduction(dim=1),
                    inducing_size=inducing_size,
                    covar_module=model.covar_module,
                    X=model.train_inputs[0],
                )
            )
            <= 20
        )

        self.assertEqual(
            len(
                select_inducing_points(
                    allocator=KMeansAllocator(dim=1),
                    inducing_size=inducing_size,
                    covar_module=model.covar_module,
                    X=model.train_inputs[0],
                )
            ),
            20,
        )
        self.assertTrue(
            len(
                select_inducing_points(
                    allocator="auto",
                    inducing_size=inducing_size,
                    covar_module=model.covar_module,
                    X=model.train_inputs[0],
                )
            )
            <= 20
        )
        self.assertTrue(
            len(
                select_inducing_points(
                    allocator=SobolAllocator(
                        bounds=torch.stack([torch.tensor([0]), torch.tensor([1])]),
                        dim=1,
                    ),
                    inducing_size=inducing_size,
                    covar_module=model.covar_module,
                    X=model.train_inputs[0],
                )
            )
            <= 20
        )
        self.assertTrue(
            len(
                select_inducing_points(
                    allocator=FixedAllocator(
                        points=torch.tensor([[0], [1], [2], [3]]), dim=1
                    ),
                    inducing_size=inducing_size,
                    covar_module=model.covar_module,
                    X=model.train_inputs[0],
                )
            )
            <= 20
        )


if __name__ == "__main__":
    unittest.main()
