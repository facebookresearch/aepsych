#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import numpy as np
import torch
from aepsych.config import Config
from aepsych.models import GPClassificationModel
from aepsych.utils import _process_bounds, get_dim, get_parameters, make_scaled_sobol


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
        dim = 1
        gridsize = 10
        mb = GPClassificationModel(lb=lb, ub=ub, dim=dim)
        grid = GPClassificationModel.dim_grid(mb, gridsize=gridsize)
        self.assertEqual(grid.shape, torch.Size([10, 1]))

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


class ParameterUtilsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_str = """
        [common]
        parnames = [par1, par2, par3]
        lb = [0, 0, 0]
        ub = [1, 1000, 10]
        choice_parnames = [par4, par5, par6, par9]
        fixed_parnames = [par7, par8]

        [par2]
        log_scale = True

        [par3]
        value_type = int

        [par4]
        choices = [a, b]

        [par5]
        choices = [x]

        [par6]
        choices = [x, y, z]
        is_ordered = True

        [par7]
        value = 123

        [par8]
        value = foo

        [par9]
        choices = [x, y, z]
        """
        self.config = Config(config_str=config_str)

    def test_get_ax_parameters(self):
        params = get_parameters(self.config)

        correct_range_params = [
            {
                "name": "par1",
                "type": "range",
                "value_type": "float",
                "log_scale": False,
                "bounds": [0.0, 1.0],
            },
            {
                "name": "par2",
                "type": "range",
                "value_type": "float",
                "log_scale": True,
                "bounds": [0.0, 1000.0],
            },
            {
                "name": "par3",
                "type": "range",
                "value_type": "int",
                "log_scale": False,
                "bounds": [0.0, 10.0],
            },
        ]
        correct_choice_params = [
            {
                "name": "par4",
                "type": "choice",
                "value_type": "str",
                "is_ordered": False,
                "values": ["a", "b"],
            },
            {
                "name": "par5",
                "type": "choice",
                "value_type": "str",
                "is_ordered": False,
                "values": ["x"],
            },
            {
                "name": "par6",
                "type": "choice",
                "value_type": "str",
                "is_ordered": True,
                "values": ["x", "y", "z"],
            },
            {
                "name": "par9",
                "type": "choice",
                "value_type": "str",
                "is_ordered": False,
                "values": ["x", "y", "z"],
            },
        ]

        correct_fixed_params = [
            {
                "name": "par7",
                "type": "fixed",
                "value": 123.0,
            },
            {
                "name": "par8",
                "type": "fixed",
                "value": "foo",
            },
        ]

        self.assertEqual(
            params, correct_range_params + correct_choice_params + correct_fixed_params
        )

    def test_get_dim(self):
        dim = get_dim(self.config)

        # 3 dims from par1, par2, par3
        # 1 binary dim from par4
        # 0 dim from par5 (effectively a fixed dim)
        # 1 dim from par6 (is_ordered makes it one continuous dim)
        # 0 dim from par7 & par8 (fixed dims aren't modeled)
        # 3 dim from par9 (one-hot vector with 3 elements)
        # 8 total dims
        self.assertEqual(8, dim)

        # Count only choice dims
        copied_config = deepcopy(self.config)
        del copied_config["common"]["parnames"]
        del copied_config["common"]["lb"]
        del copied_config["common"]["ub"]
        dim = get_dim(copied_config)
        self.assertEqual(5, dim)

        # Removing par5 does nothing
        copied_config["common"]["choice_parnames"] = "[par4, par6, par9]"
        dim = get_dim(copied_config)
        self.assertEqual(5, dim)

        # Removing par6 leaves us with 3 binary dimension and 1 continuous dimension
        copied_config["common"]["choice_parnames"] = "[par4, par9]"
        dim = get_dim(copied_config)
        self.assertEqual(4, dim)

        # Removing par9 leaves us with 1 binary dimension
        copied_config["common"]["choice_parnames"] = "[par4]"
        dim = get_dim(copied_config)
        self.assertEqual(1, dim)

        # Removing par7 & par8 does nothing
        del copied_config["common"]["fixed_parnames"]
        dim = get_dim(copied_config)
        self.assertEqual(1, dim)


if __name__ == "__main__":
    unittest.main()
