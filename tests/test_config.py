#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch
from aepsych.acquisition import MCLevelSetEstimation
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
        acqf = MCLevelSetEstimation

        [init_strat]
        generator = SobolGenerator
        min_asks = 10
        min_total_outcome_occurrences = 5

        [opt_strat]
        generator = OptimizeAcqfGenerator
        min_asks = 20
        min_post_range = 0.01
        keep_most_recent = 10

        [MCLevelSetEstimation]
        beta = 3.98
        objective = ProbitObjective

        [GPClassificationModel]
        inducing_size = 10
        mean_covar_factory = default_mean_covar_factory

        [OptimizeAcqfGenerator]
        restarts = 10
        samps = 1000
        """
        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)

        self.assertTrue(isinstance(strat.strat_list[0].generator, SobolGenerator))
        self.assertTrue(
            isinstance(strat.strat_list[1].generator, OptimizeAcqfGenerator)
        )
        self.assertTrue(isinstance(strat.strat_list[1].model, GPClassificationModel))
        self.assertTrue(strat.strat_list[1].generator.acqf is MCLevelSetEstimation)
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
        self.assertTrue(strat.strat_list[0].min_asks == 10)
        self.assertTrue(strat.strat_list[0].outcome_type == "single_probit")
        self.assertTrue(strat.strat_list[1].min_asks == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(torch.all(strat.strat_list[1].model.lb == torch.Tensor([0, 0])))
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(torch.all(strat.strat_list[1].model.ub == torch.Tensor([1, 1])))

        self.assertEqual(strat.strat_list[0].min_total_outcome_occurrences, 5)
        self.assertEqual(strat.strat_list[0].min_post_range, None)
        self.assertEqual(strat.strat_list[0].keep_most_recent, None)

        self.assertEqual(strat.strat_list[1].min_total_outcome_occurrences, 1)
        self.assertEqual(strat.strat_list[1].min_post_range, 0.01)
        self.assertEqual(strat.strat_list[1].keep_most_recent, 10)

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
        self.assertTrue(strat.strat_list[0].min_asks == 10)
        self.assertTrue(strat.strat_list[0].outcome_type == "single_probit")
        self.assertTrue(strat.strat_list[1].min_asks == 20)
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

        self.assertTrue(strat.strat_list[0].min_asks == 10)
        self.assertTrue(strat.strat_list[0].outcome_type == "single_probit")
        self.assertTrue(strat.strat_list[1].min_asks == 20)
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
        min_asks = 1

        [opt_strat1]
        generator = OptimizeAcqfGenerator
        min_asks = 1
        model = GPClassificationModel
        acqf = MCLevelSetEstimation

        [opt_strat2]
        generator = MonotonicRejectionGenerator
        min_asks = 1
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
        self.assertTrue(strat.strat_list[1].generator.acqf is MCLevelSetEstimation)

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
        config.update(config_str=config_str)
        self.assertTrue("acqf" in config["common"])
        self.assertTrue("model" in config["common"])

    def test_to_string(self):
        in_str = """
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
            min_asks = 10
            [opt_strat]
            generator = OptimizeAcqfGenerator
            min_asks = 20
            [LevelSetEstimation]
            beta = 3.98
            objective = ProbitObjective
            [GPClassificationModel]
            inducing_size = 10
            mean_covar_factory = default_mean_covar_factory
            [OptimizeAcqfGenerator]
            restarts = 10
            samps = 1000""".strip().replace(
            " ", ""
        )

        config = Config(config_str=in_str)
        out_str = str(config).strip().replace(" ", "")
        self.assertEqual(in_str, out_str)

    def test_conversion(self):
        config_str = """
        [common]
        parnames = [par1, par2]
        lb = [0, 0]
        ub = [1, 1]
        outcome_type = single_probit
        target = 0.75

        [SobolStrategy]
        n_trials = 10

        [ModelWrapperStrategy]
        n_trials = 20
        refit_every = 5

        [experiment]
        acqf = MonotonicMCLSE
        init_strat_cls = SobolStrategy
        opt_strat_cls = ModelWrapperStrategy
        modelbridge_cls = MonotonicSingleProbitModelbridge
        model = MonotonicRejectionGP

        [MonotonicMCLSE]
        beta = 3.98

        [MonotonicRejectionGP]
        inducing_size = 100
        mean_covar_factory = monotonic_mean_covar_factory

        [MonotonicSingleProbitModelbridge]
        restarts = 10
        samps = 1000
        """

        config = Config(config_str=config_str)
        self.assertEqual(config.version, "0.0")
        config.convert("0.0", "0.1")
        self.assertEqual(config.version, "0.1")

        self.assertEqual(config["common"]["strategy_names"], "[init_strat, opt_strat]")
        self.assertEqual(config["common"]["acqf"], "MonotonicMCLSE")

        self.assertEqual(config["init_strat"]["min_asks"], "10")
        self.assertEqual(config["init_strat"]["generator"], "SobolGenerator")

        self.assertEqual(config["opt_strat"]["min_asks"], "20")
        self.assertEqual(config["opt_strat"]["refit_every"], "5")
        self.assertEqual(
            config["opt_strat"]["generator"], "MonotonicRejectionGenerator"
        )
        self.assertEqual(config["opt_strat"]["model"], "MonotonicRejectionGP")

        self.assertEqual(config["MonotonicRejectionGenerator"]["restarts"], "10")
        self.assertEqual(config["MonotonicRejectionGenerator"]["samps"], "1000")

    def test_warn_about_refit(self):
        config_str = """
        [common]
        lb = [0, 0]
        ub = [1, 1]
        outcome_type = single_probit
        strategy_names = [init_strat]
        model = GPClassificationModel

        [init_strat]
        generator = SobolGenerator
        min_asks = 10
        refit_every = 5
        """

        config = Config(config_str=config_str)

        with self.assertWarns(UserWarning):
            Strategy.from_config(config, "init_strat")


if __name__ == "__main__":
    unittest.main()
