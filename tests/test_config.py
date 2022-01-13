#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch
from aepsych.acquisition.lse import LevelSetEstimation
from aepsych.acquisition.monotonic_rejection import MonotonicMCLSE
from aepsych.acquisition.objective import ProbitObjective
from aepsych.config import Config
from aepsych.generators import (
    MonotonicRejectionGenerator,
    OptimizeAcqfGenerator,
    SobolGenerator,
)
from aepsych.models import GPClassificationModel, MonotonicRejectionGP
from aepsych.strategy import SequentialStrategy, Strategy
from botorch.acquisition import qNoisyExpectedImprovement


class ConfigTestCase(unittest.TestCase):
    def test_single_probit_config(self):
        config_str = """
        [common]
        lb = [0, 0]
        ub = [1, 1]
        outcome_type = single_probit
        parnames = [par1, par2]
        strategy_names = [init_strat, opt_strat]
        model = GPClassificationModel
        acqf = LevelSetEstimation

        [init_strat]
        generator = SobolGenerator
        n_trials = 10

        [opt_strat]
        generator = OptimizeAcqfGenerator
        n_trials = 20

        [LevelSetEstimation]
        beta = 3.98
        objective = ProbitObjective

        [GPClassificationModel]
        inducing_size = 10
        mean_covar_factory = default_mean_covar_factory

        [OptimizeAcqfGenerator]
        restarts = 10
        samps = 1000

        [SobolGenerator]
        n_points = 10
        """
        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)

        self.assertTrue(isinstance(strat.strat_list[0].generator, SobolGenerator))
        self.assertTrue(
            isinstance(strat.strat_list[1].generator, OptimizeAcqfGenerator)
        )
        self.assertTrue(isinstance(strat.strat_list[1].model, GPClassificationModel))
        self.assertTrue(strat.strat_list[1].generator.acqf is LevelSetEstimation)
        # since ProbitObjective() is turned into an obj, we check for keys and then vals
        self.assertTrue(
            set(strat.strat_list[1].generator.acqf_kwargs.keys())
            == {"beta", "target", "objective"}
        )
        self.assertTrue(strat.strat_list[1].generator.acqf_kwargs["target"] == 0.75)
        self.assertTrue(strat.strat_list[1].generator.acqf_kwargs["beta"] == 3.98)
        self.assertTrue(
            isinstance(
                strat.strat_list[1].generator.acqf_kwargs["objective"],
                ProbitObjective,
            )
        )

        self.assertTrue(strat.strat_list[1].generator.restarts == 10)
        self.assertTrue(strat.strat_list[1].generator.samps == 1000)
        self.assertTrue(strat.strat_list[0].n_trials == 10)
        self.assertTrue(strat.strat_list[0].outcome_type == "single_probit")
        self.assertTrue(strat.strat_list[1].n_trials == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(torch.all(strat.strat_list[1].model.lb == torch.Tensor([0, 0])))
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(torch.all(strat.strat_list[1].model.ub == torch.Tensor([1, 1])))

    def test_missing_config_file(self):
        config_file = "../configs/does_not_exist.ini"
        config_file = os.path.join(os.path.dirname(__file__), config_file)
        with self.assertRaises(FileNotFoundError):
            Config(config_fnames=[config_file])

        with self.assertRaises(FileNotFoundError):
            Config(config_fnames=[])

    def test_monotonic_single_probit_config_file(self):
        config_file = "../configs/single_lse_example.ini"
        config_file = os.path.join(os.path.dirname(__file__), config_file)

        config = Config()
        config.update(config_fnames=[config_file])
        strat = SequentialStrategy.from_config(config)

        self.assertTrue(isinstance(strat.strat_list[0].generator, SobolGenerator))
        self.assertTrue(strat.strat_list[0].model is None)

        self.assertTrue(
            isinstance(strat.strat_list[1].generator, MonotonicRejectionGenerator)
        )
        self.assertTrue(strat.strat_list[1].generator.acqf is MonotonicMCLSE)
        self.assertTrue(
            set(strat.strat_list[1].generator.acqf_kwargs.keys())
            == {"beta", "target", "objective"}
        )
        self.assertTrue(strat.strat_list[1].generator.acqf_kwargs["target"] == 0.75)
        self.assertTrue(strat.strat_list[1].generator.acqf_kwargs["beta"] == 3.98)
        self.assertTrue(
            isinstance(
                strat.strat_list[1].generator.acqf_kwargs["objective"],
                ProbitObjective,
            )
        )
        self.assertTrue(
            strat.strat_list[1].generator.model_gen_options["raw_samples"] == 1000
        )
        self.assertTrue(strat.strat_list[0].n_trials == 10)
        self.assertTrue(strat.strat_list[0].outcome_type == "single_probit")
        self.assertTrue(strat.strat_list[1].n_trials == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(torch.all(strat.strat_list[1].model.lb == torch.Tensor([0, 0])))
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(torch.all(strat.strat_list[1].model.ub == torch.Tensor([1, 1])))

    def test_nonmonotonic_optimization_config_file(self):
        config_file = "../configs/nonmonotonic_optimization_example.ini"
        config_file = os.path.join(os.path.dirname(__file__), config_file)

        config = Config()
        config.update(config_fnames=[config_file])
        strat = SequentialStrategy.from_config(config)

        self.assertTrue(isinstance(strat.strat_list[0].generator, SobolGenerator))
        self.assertTrue(strat.strat_list[0].model is None)

        self.assertTrue(
            isinstance(strat.strat_list[1].generator, OptimizeAcqfGenerator)
        )
        self.assertTrue(strat.strat_list[1].generator.acqf is qNoisyExpectedImprovement)
        self.assertTrue(
            set(strat.strat_list[1].generator.acqf_kwargs.keys()) == {"objective"}
        )
        self.assertTrue(
            isinstance(
                strat.strat_list[1].generator.acqf_kwargs["objective"],
                ProbitObjective,
            )
        )

        self.assertTrue(strat.strat_list[0].n_trials == 10)
        self.assertTrue(strat.strat_list[0].outcome_type == "single_probit")
        self.assertTrue(strat.strat_list[1].n_trials == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(torch.all(strat.strat_list[1].model.lb == torch.Tensor([0, 0])))
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(torch.all(strat.strat_list[1].model.ub == torch.Tensor([1, 1])))

    def test_name_conflict_warns(self):
        class DummyMod:
            pass

        Config.register_object(DummyMod)

        with self.assertWarns(Warning):
            Config.register_object(DummyMod)

    def test_sobol_n_trials(self):
        for n_trials in [-1, 0, 1]:
            config_str = f"""
            [common]
            lb = [0]
            ub = [1]
            parnames = [par1]
            strategy_names = [init_strat]

            [init_strat]
            generator = SobolGenerator
            n_trials = {n_trials}
            """
            config = Config()
            config.update(config_str=config_str)
            strat = Strategy.from_config(config, "init_strat")
            self.assertEqual(strat.n_trials, n_trials)
            self.assertEqual(strat.finished, n_trials <= 0)

    def test_multiple_models_and_strats(self):
        config_str = """
        [common]
        lb = [0, 0]
        ub = [1, 1]
        outcome_type = single_probit
        parnames = [par1, par2]
        strategy_names = [init_strat, opt_strat1, opt_strat2]

        [init_strat]
        generator = SobolGenerator
        n_trials = 1

        [opt_strat1]
        generator = OptimizeAcqfGenerator
        n_trials = 1
        model = GPClassificationModel
        acqf = LevelSetEstimation

        [opt_strat2]
        generator = MonotonicRejectionGenerator
        n_trials = 1
        model = MonotonicRejectionGP
        acqf = MonotonicMCLSE
        """

        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)

        self.assertTrue(isinstance(strat.strat_list[0].generator, SobolGenerator))
        self.assertTrue(strat.strat_list[0].model is None)

        self.assertTrue(
            isinstance(strat.strat_list[1].generator, OptimizeAcqfGenerator)
        )
        self.assertTrue(isinstance(strat.strat_list[1].model, GPClassificationModel))
        self.assertTrue(strat.strat_list[1].generator.acqf is LevelSetEstimation)

        self.assertTrue(
            isinstance(strat.strat_list[2].generator, MonotonicRejectionGenerator)
        )
        self.assertTrue(isinstance(strat.strat_list[2].model, MonotonicRejectionGP))
        self.assertTrue(strat.strat_list[2].generator.acqf is MonotonicMCLSE)

    def test_experiment_deprecation(self):
        config_str = """
            [experiment]
            acqf = PairwiseMCPosteriorVariance
            model = PairwiseProbitModel
            """
        config = Config()
        with self.assertWarns(DeprecationWarning):
            config.update(config_str=config_str)
        self.assertTrue("acqf" in config["common"])
        self.assertTrue("model" in config["common"])
