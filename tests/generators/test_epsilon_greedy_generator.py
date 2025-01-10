#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from aepsych.acquisition import MonotonicMCLSE
from aepsych.config import Config
from aepsych.generators import EpsilonGreedyGenerator, MonotonicRejectionGenerator


class TestEpsilonGreedyGenerator(unittest.TestCase):
    def test_epsilon_greedy(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        total_trials = 2000
        extra_acqf_args = {"target": 0.75, "beta": 1.96}
        lb = torch.tensor([0.0])
        ub = torch.tensor([1.0])

        for epsilon in (0.1, 0.5):
            gen = EpsilonGreedyGenerator(
                subgenerator=MonotonicRejectionGenerator(
                    acqf=MonotonicMCLSE, acqf_kwargs=extra_acqf_args, lb=lb, ub=ub
                ),
                epsilon=epsilon,
                lb=lb,
                ub=ub,
            )
            model = MagicMock()
            gen.subgenerator.gen = MagicMock()
            for _ in range(total_trials):
                gen.gen(1, model)

            self.assertTrue(
                np.abs(gen.subgenerator.gen.call_count / total_trials - (1 - epsilon))
                < 0.01
            )

    def test_epsilon_greedy_fixed(self):
        extra_acqf_args = {"target": 0.75, "beta": 1.96}
        lb = torch.tensor([0.0])
        ub = torch.tensor([1.0])

        gen = EpsilonGreedyGenerator(
            subgenerator=MonotonicRejectionGenerator(
                acqf=MonotonicMCLSE, acqf_kwargs=extra_acqf_args, lb=lb, ub=ub
            ),
            epsilon=1,
            lb=lb,
            ub=ub,
        )
        model = MagicMock()
        gen.subgenerator.gen = MagicMock()
        result = gen.gen(1, model, fixed_features={0: 0.5})
        self.assertTrue(result == 0.5)

    def test_greedyepsilon_config(self):
        config_str = """
            [common]
            acqf = MonotonicMCLSE
            lb = [0]
            ub = [1]
            [EpsilonGreedyGenerator]
            subgenerator = MonotonicRejectionGenerator
            epsilon = .5
            """
        config = Config()
        config.update(config_str=config_str)
        gen = EpsilonGreedyGenerator.from_config(config)
        self.assertIsInstance(gen.subgenerator, MonotonicRejectionGenerator)
        self.assertEqual(gen.subgenerator.acqf, MonotonicMCLSE)
        self.assertEqual(gen.epsilon, 0.5)
