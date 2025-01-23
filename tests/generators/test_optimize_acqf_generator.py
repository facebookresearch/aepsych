#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import unittest

import numpy as np
import torch
from aepsych.acquisition import EAVC, MCLevelSetEstimation
from aepsych.acquisition.objective import ProbitObjective
from aepsych.config import Config
from aepsych.generators import OptimizeAcqfGenerator
from aepsych.models import GPClassificationModel, PairwiseProbitModel
from aepsych.strategy import SequentialStrategy, Strategy
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from sklearn.datasets import make_classification


class TestOptimizeAcqfGenerator(unittest.TestCase):
    def test_time_limits(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)

        X, y = make_classification(
            n_samples=100,
            n_features=8,
            n_redundant=3,
            n_informative=5,
            random_state=1,
            n_clusters_per_class=4,
        )
        X, y = torch.Tensor(X), torch.Tensor(y)
        lb = -3 * torch.ones(8)
        ub = 3 * torch.ones(8)
        inducing_size = 10
        bounds = torch.stack([lb, ub])

        model = GPClassificationModel(
            dim=8,
            max_fit_time=0.5,
            inducing_size=inducing_size,
        )

        model.fit(X, y)
        generator = OptimizeAcqfGenerator(
            acqf=MCLevelSetEstimation,
            acqf_kwargs={"beta": 1.96, "target": 0.5},
            lb=lb,
            ub=ub,
        )

        start = time.time()
        generator.gen(1, model)
        end = time.time()
        long = end - start
        generator = OptimizeAcqfGenerator(
            acqf=MCLevelSetEstimation,
            acqf_kwargs={"beta": 1.96, "target": 0.5},
            max_gen_time=0.1,
            lb=lb,
            ub=ub,
        )

        start = time.time()
        generator.gen(1, model)
        end = time.time()
        short = end - start

        # very loose test because fit time is only approximately computed
        # on very fast machines, short sometimes actually wins, but hopefully not by
        # much
        self.assertTrue(long - short > -0.05, f"Long time: {long}, short time: {short}")

    def test_instantiate_eubo(self):
        config = """
        [common]
        lb = [-1]
        ub = [1]

        [OptimizeAcqfGenerator]
        acqf = AnalyticExpectedUtilityOfBestOption
        stimuli_per_trial = 2
        """
        generator = OptimizeAcqfGenerator.from_config(Config(config_str=config))
        self.assertTrue(generator.acqf == AnalyticExpectedUtilityOfBestOption)

        # need a fitted model in order to instantiate the acqf successfully
        model = PairwiseProbitModel(lb=[-1], ub=[1])
        train_x = torch.Tensor([-0.5, 1, 0.5, -1]).reshape((2, 1, 2))
        train_y = torch.Tensor([0, 1])
        model.fit(train_x, train_y)
        acqf = generator._instantiate_acquisition_fn(model=model)
        self.assertTrue(isinstance(acqf, AnalyticExpectedUtilityOfBestOption))

    def test_batch_generation(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)

        X, y = make_classification(
            n_samples=100,
            n_features=3,
            n_redundant=0,
            n_informative=3,
            random_state=1,
            n_clusters_per_class=4,
        )
        X, y = torch.Tensor(X), torch.Tensor(y)
        lb = -3 * torch.ones(3)
        ub = 3 * torch.ones(3)
        inducing_size = 10

        model = GPClassificationModel(
            dim=3,
            max_fit_time=0.5,
            inducing_size=inducing_size,
        )

        model.fit(X, y)

        generator = OptimizeAcqfGenerator(
            acqf=qLogNoisyExpectedImprovement,
            acqf_kwargs={"objective": ProbitObjective()},
            lb=lb,
            ub=ub,
        )
        cands = generator.gen(2, model)
        self.assertEqual(len(cands), 2)

        generator = OptimizeAcqfGenerator(
            acqf=MCLevelSetEstimation,
            acqf_kwargs={"objective": ProbitObjective()},
            lb=lb,
            ub=ub,
        )
        cands = generator.gen(2, model)
        self.assertEqual(len(cands), 2)

    def test_batch_generation_only_one(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)

        X, y = make_classification(
            n_samples=100,
            n_features=3,
            n_redundant=0,
            n_informative=3,
            random_state=1,
            n_clusters_per_class=4,
        )
        X, y = torch.Tensor(X), torch.Tensor(y)
        lb = -3 * torch.ones(3)
        ub = 3 * torch.ones(3)
        inducing_size = 10

        model = GPClassificationModel(
            dim=3,
            max_fit_time=0.5,
            inducing_size=inducing_size,
        )

        model.fit(X, y)

        generator = OptimizeAcqfGenerator(
            acqf=EAVC,
            acqf_kwargs={"target": 0.75, "lb": lb, "ub": ub},
            lb=lb,
            ub=ub,
        )

        with self.assertWarns(UserWarning) as cm:
            cands = generator.gen(2, model)
            self.assertTrue(
                "can only generate one point at a time"
                in cm.warnings[0].message.args[0]
            )

        self.assertEqual(len(cands), 1)

    def test_get_acqf_options(self):
        config_str = """
            [common]
            parnames = [signal1]
            stimuli_per_trial = 1
            outcome_types = [binary]
            target = 0.75
            strategy_names = [init_strat, opt_strat]

            [signal1]
            par_type = continuous
            lower_bound = 1
            upper_bound = 100

            [init_strat]
            min_total_tells = 10
            generator = SobolGenerator

            [SobolGenerator]
            seed = 12345

            [opt_strat]
            min_total_tells = 1
            generator = OptimizeAcqfGenerator
            model = GPClassificationModel
            acqf = EAVC
        """
        config = Config()
        config.update(config_str=config_str)

        strat = Strategy.from_config(config, "opt_strat")

        # We expect ub/lb to be found and they're in the transformed space
        acqf_kwargs = strat.generator._get_acqf_options(strat.generator.acqf, config)
        self.assertTrue("lb" in acqf_kwargs)
        self.assertTrue(acqf_kwargs["lb"] == 0)
        self.assertTrue("ub" in acqf_kwargs)
        self.assertTrue(acqf_kwargs["ub"] == 1)

    def test_acqf_from_config(self):
        n_init = 10
        n_opt = 1
        config_str = f"""
            [common]
            parnames = [signal1]
            stimuli_per_trial = 1
            outcome_types = [binary]
            target = 0.75
            strategy_names = [init_strat, opt_strat]

            [signal1]
            par_type = continuous
            lower_bound = 100
            upper_bound = 200

            [init_strat]
            min_total_tells = {n_init}
            generator = SobolGenerator

            [SobolGenerator]
            seed = 12345

            [opt_strat]
            min_total_tells = {n_opt}
            generator = OptimizeAcqfGenerator
            model = GPClassificationModel
            acqf = EAVC

            [OptimizeAcqfGenerator]
            samps = 10

            [EAVC]
            target = 0.75
        """
        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)

        for _ in range(n_init + n_opt):
            point = strat.gen()
            response = float(torch.randint(100, 200, size=(1,)) > point)
            strat.add_data(point, response)

        # Last generator should be EAVC, needs to have the right bounds
        self.assertTrue(strat.generator.acqf_kwargs["lb"] == 0)
        self.assertTrue(strat.generator.acqf_kwargs["ub"] == 1)


if __name__ == "__main__":
    unittest.main()
