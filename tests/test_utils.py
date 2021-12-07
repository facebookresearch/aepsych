#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import numpy as np
import torch
from aepsych.utils import make_scaled_sobol
from aepsych.models import GPClassificationModel


class TestSequenceGenerators(unittest.TestCase):
    def test_scaled_sobol_asserts(self):

        lb = np.r_[0, 0, 1]
        ub = np.r_[1]
        with self.assertRaises(AssertionError):
            make_scaled_sobol(lb, ub, 10)

    def test_scaled_sobol_sizes(self):
        lb = np.r_[0, 1]
        ub = np.r_[1, 30]
        grid = make_scaled_sobol(lb, ub, 100)
        self.assertEqual(grid.shape, (100, 2))

    def test_dim_grid_model_size(self):

        lb = -4.0
        ub = 4.0
        dim = 1
        gridsize = 10
        mb = GPClassificationModel(lb=lb, ub=ub, dim=dim)
        grid = GPClassificationModel.dim_grid(mb, gridsize=gridsize)
        self.assertEqual(grid.shape, torch.Size([10, 1]))


if __name__ == "__main__":
    unittest.main()
