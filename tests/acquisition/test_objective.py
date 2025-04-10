#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import product

import numpy as np
import numpy.testing as npt
import torch
from aepsych.acquisition.objective import (
    AffinePosteriorTransform,
    FloorGumbelObjective,
    FloorLogitObjective,
    FloorProbitObjective,
)
from aepsych.config import Config
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


class AffinePosteriorTransformTests(unittest.TestCase):
    def test_from_config(self):
        config_str = """
        [AffinePosteriorTransform]
        weights = [1, -1]
        offset = 1
        """
        config = Config(config_str=config_str)
        apt = AffinePosteriorTransform.from_config(config=config)
        npt.assert_array_equal(apt.weights, torch.tensor([1.0, -1.0]))
        self.assertEqual(apt.offset, 1.0)

    def test_defaults(self):
        config_str = """
            [common]
            parnames = [foo]
            outcome_names = [baz, qux]
            outcome_types = [binary, continuous]
            stimuli_per_trial = 1
            strategy_names = [init_strat, opt_strat]

            [foo]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = SobolGenerator

            [opt_strat]
            generator = OptimizeAcqfGenerator
            model = GPClassificationModel

            [OptimizeAcqfGenerator]
            acqf = qLogNoisyExpectedImprovement

            [qLogNoisyExpectedImprovement]
            posterior_transform = AffinePosteriorTransform
        """
        config = Config(config_str=config_str)
        apt = AffinePosteriorTransform.from_config(config=config)
        npt.assert_array_equal(apt.weights, torch.tensor([0.5, 0.5]))
        self.assertEqual(apt.offset, 0.0)
