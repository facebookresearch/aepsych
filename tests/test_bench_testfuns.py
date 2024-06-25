#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from aepsych.benchmark.test_functions import make_songetal_testfun
from aepsych.utils import dim_grid


class BenchmarkTestCase(unittest.TestCase):
    def test_songetal_funs_smoke(self):
        valid_phenotypes = ["Metabolic", "Sensory", "Metabolic+Sensory", "Older-normal"]
        grid = dim_grid(lower=[-3, -20], upper=[4, 120], gridsize=30)
        try:
            for phenotype in valid_phenotypes:
                testfun = make_songetal_testfun(phenotype=phenotype)
                f = testfun(grid)
                self.assertTrue(f.shape == torch.Size([900]))
        except Exception:
            self.fail()

        with self.assertRaises(AssertionError):
            _ = make_songetal_testfun(phenotype="not_a_real_phenotype")
