#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from aepsych import Config
from aepsych.models import IndependentGPsModel
from aepsych.models.utils import dim_grid
from aepsych.strategy import SequentialStrategy


def f_2d(x, target=None):
    """
    Distance to target
    """
    if target is None:
        target = torch.tensor([0.0, 0.0])

    target = target.to(x)
    if x.ndim > 1:
        return torch.exp(-torch.linalg.vector_norm(x - target, dim=1))

    return torch.exp(-torch.linalg.vector_norm(x - target))


class IndependentGPStratTest(unittest.TestCase):
    def test_end_to_end(self):
        torch.manual_seed(1)
        np.random.seed(1)

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
            min_asks = 150

            [opt_strat]
            generator = IndependentOptimizeAcqfGenerator
            model = IndependentGPsModel
            min_asks = 1

            [IndependentOptimizeAcqfGenerator]
            generators = [BazGen, QuxGen]
            use_gpu = True

            [BazGen]
            class = OptimizeAcqfGenerator
            acqf = EAVC

            [EAVC]
            target = 0.75

            [QuxGen]
            class = OptimizeAcqfGenerator
            acqf = qLogNoisyExpectedImprovement

            [IndependentGPsModel]
            models = [model1, model2]
            use_gpu = True

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

                strat.add_data(
                    point.cpu(), torch.tensor([[baz_response, qux_response]])
                )

        x_grid = dim_grid(lower=strat.lb, upper=strat.ub)
        pred_y = strat.model.predict(x_grid)

        norm = torch.distributions.Normal(0, 1)
        baz_target = f_2d(
            x_grid[np.argmin((norm.cdf(pred_y[0][:, 0].cpu()) - 0.75) ** 2)]
        )
        qux_max = x_grid[torch.argmax(pred_y[0][:, 1])]

        # Pretty wide check on binary, but that's the nature of it
        self.assertLessEqual(torch.abs(baz_target - 0.75), 0.15)
        self.assertTrue(torch.all(torch.abs(qux_max - -0.5) < 0.1))

        self.assertEqual(strat.model.device.type, "cuda")

    def test_move_independent_models(self):
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

            [IndependentGPsModel]
            models = [model1, model2]
            use_gpu = True

            [model1]
            class = GPClassificationModel

            [model2]
            class = GPRegressionModel
        """
        config = Config(config_str=config_str)
        model = IndependentGPsModel.from_config(config)
        model

        self.assertEqual(model.device.type, "cpu")

        model.cuda()

        self.assertEqual(model.device.type, "cuda")
        self.assertEqual(model[0].device.type, "cuda")
        self.assertEqual(model[1].device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
