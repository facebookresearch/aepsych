#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import unittest

import numpy as np
import torch
from aepsych.acquisition import MCLevelSetEstimation
from aepsych.config import Config
from aepsych.generators import AxOptimizeAcqfGenerator, OptimizeAcqfGenerator
from aepsych.models import (
    ContinuousRegressionGP,
    GPClassificationModel,
    PairwiseProbitModel,
)
from ax.modelbridge import Models
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

        model = GPClassificationModel(
            lb=-3 * torch.ones(8),
            ub=3 * torch.ones(8),
            max_fit_time=0.5,
            inducing_size=10,
        )

        model.fit(X, y)
        generator = OptimizeAcqfGenerator(
            acqf=MCLevelSetEstimation, acqf_kwargs={"beta": 1.96, "target": 0.5}
        )

        start = time.time()
        generator.gen(1, model)
        end = time.time()
        long = end - start
        generator = OptimizeAcqfGenerator(
            acqf=MCLevelSetEstimation,
            acqf_kwargs={"beta": 1.96, "target": 0.5},
            max_gen_time=0.1,
        )

        start = time.time()
        generator.gen(1, model)
        end = time.time()
        short = end - start

        # very loose test because fit time is only approximately computed
        self.assertTrue(long > short)

    def test_instantiate_eubo(self):
        config = """
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

    def test_axoptimizeacqf_config(self):
        config_str = """
                [common]
                use_ax = True
                parnames = [foo]
                lb = [0]
                ub = [1]
                stimuli_per_trial = 1
                outcome_types = [continuous]
                strat_names = [opt]

                [opt]
                generator = OptimizeAcqfGenerator
                model = ContinuousRegressionGP

                [OptimizeAcqfGenerator]
                acqf = MCLevelSetEstimation
                max_gen_time = 0.1
                num_restarts = 1
                raw_samples = 100

                [MCLevelSetEstimation]
                beta = 1
                target = 0.5
                """

        config = Config(config_str=config_str)
        gen = AxOptimizeAcqfGenerator.from_config(config, "opt")

        self.assertEqual(gen.model, Models.BOTORCH_MODULAR)

        self.assertEqual(
            gen.model_kwargs["surrogate"].botorch_model_class, ContinuousRegressionGP
        )
        self.assertEqual(
            gen.model_gen_kwargs["model_gen_options"]["optimizer_kwargs"][
                "num_restarts"
            ],
            1,
        )
        self.assertEqual(
            gen.model_gen_kwargs["model_gen_options"]["optimizer_kwargs"][
                "raw_samples"
            ],
            100,
        )
        self.assertEqual(
            gen.model_gen_kwargs["model_gen_options"]["optimizer_kwargs"][
                "timeout_sec"
            ],
            0.1,
        )
        self.assertEqual(gen.model_kwargs["acquisition_options"]["target"], 0.5)
        self.assertEqual(gen.model_kwargs["acquisition_options"]["beta"], 1.0)
        # TODO: Implement max_gen_time


if __name__ == "__main__":
    unittest.main()
