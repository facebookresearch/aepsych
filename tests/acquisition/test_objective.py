#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import product

import numpy as np
import torch
from aepsych.acquisition.objective import (
    FloorGumbelObjective,
    FloorLogitObjective,
    FloorProbitObjective,
)
from parameterized import parameterized
from scipy.stats import gumbel_l, logistic, norm

objective_pairs = [
    (FloorLogitObjective, logistic),
    (FloorProbitObjective, norm),
    (FloorGumbelObjective, gumbel_l),
]
floors = [0, 0.5, 0.33]
all_tests = list(product(objective_pairs, floors))


class FloorLinkTests(unittest.TestCase):
    @parameterized.expand(all_tests)
    def test_floor_links(self, objectives, floor):
        our_objective, scipy_dist = objectives
        x = np.linspace(-3, 3, 50)

        scipy_answer = scipy_dist.cdf(x)
        scipy_answer = scipy_answer * (1 - floor) + floor

        our_link = our_objective(floor=floor)
        our_answer = our_link(torch.Tensor(x).unsqueeze(-1))
        self.assertTrue(np.allclose(scipy_answer, our_answer.numpy()))

        our_inverse = our_link.inverse(our_answer)
        self.assertTrue(np.allclose(x, our_inverse.numpy()))
