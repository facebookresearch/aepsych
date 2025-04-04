#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from aepsych import Config
from aepsych.strategy import SequentialStrategy


def f_2d(x, target=None):
    """
    Distance to target
    """
    if target is None:
        target = torch.tensor([0.0, 0.0])

    return torch.exp(-torch.linalg.vector_norm(x - target))


class IndependentGPStratTest(unittest.TestCase):
    def test_non_acqf_gen_smoketest(self):
        config_str = """
            [common]
            parnames = [foo, bar]
            outcome_names = [baz, qux]
            outcome_types = [binary, continuous]
            stimuli_per_trial = 1
            strategy_names = [init_strat, opt_strat]

            [foo]
            par_type = continuous
            lower_bound = -2
            upper_bound = 2

            [bar]
            par_type = continuous
            lower_bound = -2
            upper_bound = 2

            [init_strat]
            generator = SobolGenerator
            min_asks = 10

            [opt_strat]
            generator = IndependentOptimizeAcqfGenerator
            model = IndependentGPsModel
            min_asks = 1

            [IndependentOptimizeAcqfGenerator]
            generators = [BazGen, QuxGen]

            [BazGen]
            class = SobolGenerator
            seed = 1

            [QuxGen]
            class = OptimizeAcqfGenerator
            acqf = qLogNoisyExpectedImprovement

            [IndependentGPsModel]
            models = [model1, model2]

            [model1]
            class = GPClassificationModel

            [model2]
            class = GPRegressionModel
        """
        config = Config(config_str=config_str)
        strat = SequentialStrategy.from_config(config)
        while not strat.finished:
            points = strat.gen(1)

            for point in points:
                baz_response = torch.bernoulli(f_2d(point))
                qux_response = f_2d(point, target=torch.tensor([-0.5, -0.5]))

                strat.add_data(point, torch.tensor([[baz_response, qux_response]]))

    def test_diff_max_asks(self):
        config_str = """
            [common]
            parnames = [foo, bar]
            outcome_names = [baz, qux]
            outcome_types = [binary, continuous]
            stimuli_per_trial = 1
            strategy_names = [init_strat, opt_strat]

            [foo]
            par_type = continuous
            lower_bound = -2
            upper_bound = 2

            [bar]
            par_type = continuous
            lower_bound = -2
            upper_bound = 2

            [init_strat]
            generator = SobolGenerator
            min_asks = 10

            [opt_strat]
            generator = IndependentOptimizeAcqfGenerator
            model = IndependentGPsModel
            min_asks = 2

            [IndependentOptimizeAcqfGenerator]
            generators = [BazGen, QuxGen]

            [BazGen]
            class = ManualGenerator
            points = [[-1.2345, 1.2345], [0, 0]]
            min_asks = 2

            [QuxGen]
            class = OptimizeAcqfGenerator
            acqf = qLogNoisyExpectedImprovement

            [IndependentGPsModel]
            models = [model1, model2]

            [model1]
            class = GPClassificationModel

            [model2]
            class = GPRegressionModel
        """
        config = Config(config_str=config_str)
        with self.assertRaises(ValueError):
            SequentialStrategy.from_config(config)


if __name__ == "__main__":
    unittest.main()
