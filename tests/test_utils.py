#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from aepsych.utils import _process_bounds, dim_grid, make_scaled_sobol


class UtilsTestCase(unittest.TestCase):
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
        gridsize = 10

        grid = dim_grid(
            lower=torch.tensor([lb]), upper=torch.tensor([ub]), gridsize=gridsize
        )
        self.assertEqual(grid.shape, torch.Size([10, 1]))

    def test_dim_grid_slice(self):
        lb = torch.tensor([0, 0, 0])
        ub = torch.tensor([1, 1, 1])
        grid = dim_grid(lb, ub, slice_dims={1: 0.5})

        self.assertTrue(np.all(grid.shape == (900, 3)))

    def test_dim_grid_values(self):
        """Test that the grid points are correctly spaced."""
        lb = torch.tensor([0.0, 1.0])
        ub = torch.tensor([1.0, 3.0])
        gridsize = 5

        grid = dim_grid(lb, ub, gridsize=gridsize)

        # Check shape
        self.assertEqual(grid.shape, (25, 2))

        # Check values for first dimension
        expected_x_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        x_values = torch.unique(grid[:, 0])
        self.assertTrue(torch.allclose(x_values, expected_x_values))

        # Check values for second dimension
        expected_y_values = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])
        y_values = torch.unique(grid[:, 1])
        self.assertTrue(torch.allclose(y_values, expected_y_values))

    def test_process_bounds(self):
        lb, ub, dim = _process_bounds(np.r_[0, 1], np.r_[2, 3], None)
        self.assertTrue(torch.all(lb == torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.all(ub == torch.tensor([2.0, 3.0])))
        self.assertEqual(dim, 2)

        # Wrong dim
        with self.assertRaises(AssertionError):
            _process_bounds(np.r_[0, 0], np.r_[1, 1], 3)

        # ub < lb
        with self.assertRaises(AssertionError):
            _process_bounds(np.r_[1], np.r_[0], None)


if __name__ == "__main__":
    unittest.main()
