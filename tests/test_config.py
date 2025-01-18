#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import unittest
import uuid
from pathlib import Path

import torch
from aepsych.acquisition import EAVC, MCLevelSetEstimation
from aepsych.acquisition.monotonic_rejection import MonotonicMCLSE
from aepsych.acquisition.objective import FloorGumbelObjective, ProbitObjective
from aepsych.config import Config, ParameterConfigError
from aepsych.generators import (
    MonotonicRejectionGenerator,
    OptimizeAcqfGenerator,
    SobolGenerator,
)
from aepsych.likelihoods import BernoulliObjectiveLikelihood
from aepsych.models import (
    GPClassificationModel,
    HadamardSemiPModel,
    MonotonicRejectionGP,
    PairwiseProbitModel,
)
from aepsych.models.inducing_points import SobolAllocator
from aepsych.server import AEPsychServer
from aepsych.server.message_handlers.handle_setup import configure
from aepsych.strategy import SequentialStrategy, Strategy
from aepsych.transforms import ParameterTransforms, transform_options
from aepsych.transforms.ops import Log10Plus, NormalizeScale
from aepsych.version import __version__
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.acquisition.active_learning import PairwiseMCPosteriorVariance


class ConfigTestCase(unittest.TestCase):
    def test_single_probit_config(self):
        config_str = """
        [common]
        stimuli_per_trial = 1
        outcome_types = [binary]
        parnames = [par1, par2]
        strategy_names = [init_strat, opt_strat]
        model = GPClassificationModel
        acqf = MCLevelSetEstimation

        [par1]
        par_type = continuous
        lower_bound = 1
        upper_bound = 10

        [par2]
        par_type = continuous
        lower_bound = -1
        upper_bound = 1

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
        target = 0.75
        beta = 3.84
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

        self.assertTrue(
            config.get_section("MCLevelSetEstimation")
            == {"beta": "3.84", "objective": "ProbitObjective", "target": "0.75"}
        )
        self.assertTrue(
            config.get_section("OptimizeAcqfGenerator")
            == {"restarts": "10", "samps": "1000"}
        )
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
        self.assertTrue(strat.strat_list[1].generator.acqf_kwargs["beta"] == 3.84)
        self.assertTrue(
            isinstance(
                strat.strat_list[1].generator.acqf_kwargs["objective"],
                ProbitObjective,
            )
        )

        self.assertTrue(strat.strat_list[1].generator.restarts == 10)
        self.assertTrue(strat.strat_list[1].generator.samps == 1000)
        self.assertTrue(strat.strat_list[0].min_asks == 10)
        self.assertTrue(strat.strat_list[0].stimuli_per_trial == 1)
        self.assertTrue(strat.strat_list[0].outcome_types == ["binary"])
        self.assertTrue(strat.strat_list[1].min_asks == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(
            torch.all(
                strat.transforms.untransform(strat.strat_list[1].generator.lb)
                == torch.Tensor([1, -1])
            )
        )
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(
            torch.all(
                strat.transforms.untransform(strat.strat_list[1].generator.ub)
                == torch.Tensor([10, 1])
            )
        )
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

    def test_single_probit_config_file(self):
        config_file = "../configs/single_lse_example.ini"
        config_file = os.path.join(os.path.dirname(__file__), config_file)

        config = Config()
        config.update(config_fnames=[config_file])
        strat = SequentialStrategy.from_config(config)

        self.assertTrue(isinstance(strat.strat_list[0].generator, SobolGenerator))
        self.assertTrue(strat.strat_list[0].model is None)

        self.assertTrue(
            isinstance(strat.strat_list[1].generator, OptimizeAcqfGenerator)
        )
        self.assertTrue(strat.strat_list[1].generator.acqf is EAVC)
        self.assertTrue(
            set(strat.strat_list[1].generator.acqf_kwargs.keys())
            == {"lb", "ub", "target"}
        )
        self.assertTrue(strat.strat_list[1].generator.acqf_kwargs["target"] == 0.75)
        self.assertTrue(strat.strat_list[1].generator.samps == 1000)
        self.assertTrue(strat.strat_list[0].min_asks == 10)
        self.assertTrue(strat.strat_list[0].stimuli_per_trial == 1)
        self.assertTrue(strat.strat_list[0].outcome_types == ["binary"])
        self.assertTrue(strat.strat_list[1].min_asks == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(
            torch.all(strat.strat_list[1].generator.lb == torch.Tensor([0, 0]))
        )
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(
            torch.all(strat.strat_list[1].generator.ub == torch.Tensor([1, 1]))
        )

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
        self.assertTrue(
            strat.strat_list[1].generator.acqf is qLogNoisyExpectedImprovement
        )
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
        self.assertTrue(strat.strat_list[0].stimuli_per_trial == 1)
        self.assertTrue(strat.strat_list[0].outcome_types == ["binary"])
        self.assertTrue(strat.strat_list[1].min_asks == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(
            torch.all(strat.strat_list[1].generator.lb == torch.Tensor([0, 0]))
        )
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(
            torch.all(strat.strat_list[1].generator.ub == torch.Tensor([1, 1]))
        )

    def test_name_conflict_warns(self):
        class DummyMod:
            pass

        Config.register_object(DummyMod)

        with self.assertWarns(Warning):
            Config.register_object(DummyMod)

    def test_multiple_models_and_strats(self):
        config_str = """
        [common]
        stimuli_per_trial = 1
        outcome_types = [binary]
        parnames = [par1, par2]
        strategy_names = [init_strat, opt_strat1, opt_strat2]

        [par1]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [par2]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

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
            stimuli_per_trial = 1
            outcome_types = [binary]
            parnames = [par1, par2]
            strategy_names = [init_strat, opt_strat]
            model = GPClassificationModel
            acqf = LevelSetEstimation
            lb = [0, 0]
            ub = [1, 1]
            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1
            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1
            [init_strat]
            generator = SobolGenerator
            min_asks = 10
            [opt_strat]
            generator = OptimizeAcqfGenerator
            min_asks = 20
            [LevelSetEstimation]
            beta = 3.84
            objective = ProbitObjective
            [GPClassificationModel]
            inducing_size = 10
            mean_covar_factory = default_mean_covar_factory
            [OptimizeAcqfGenerator]
            restarts = 10
            samps = 1000""".strip().replace(" ", "")

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
        beta = 3.84

        [MonotonicRejectionGP]
        inducing_size = 100
        mean_covar_factory = monotonic_mean_covar_factory

        [MonotonicSingleProbitModelbridge]
        restarts = 10
        samps = 1000
        """

        config = Config(config_str=config_str)
        self.assertEqual(config.version, "0.0")
        config.convert_to_latest()
        self.assertEqual(config.version, __version__)

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

        self.assertEqual(config["common"]["stimuli_per_trial"], "1")
        self.assertEqual(config["common"]["outcome_types"], "[binary]")

    def test_warn_about_refit(self):
        config_str = """
        [common]
        parnames = [par1, par2]
        stimuli_per_trial = 1
        outcome_types = [binary]
        strategy_names = [init_strat]
        model = GPClassificationModel

        [par1]
        par_type = continuous
        lower_bound = 0 # lower bound
        upper_bound = 1 # upper bound

        [par2]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [init_strat]
        generator = SobolGenerator
        min_asks = 10
        refit_every = 5
        """

        config = Config(config_str=config_str)

        with self.assertWarns(UserWarning):
            Strategy.from_config(config, "init_strat")

    def test_nested_tensor(self):
        points = [[0.25, 0.75], [0.5, 0.9]]
        config_str = f"""
                [common]
                parnames = [par1, par2]

                [par1]
                par_type = continuous
                lower_bound = 0
                upper_bound = 1

                [par2]
                par_type = continuous
                lower_bound = 0
                upper_bound = 1

                [SampleAroundPointsGenerator]
                points = {points}
        """
        config = Config()
        config.update(config_str=config_str)

        config_points = config.gettensor("SampleAroundPointsGenerator", "points")
        self.assertTrue(torch.all(config_points == torch.tensor(points)))

    def test_pairwise_probit_config(self):
        config_str = """
            [common]
            stimuli_per_trial = 2
            outcome_types = [binary]
            parnames = [par1, par2]
            strategy_names = [init_strat, opt_strat]
            acqf = PairwiseMCPosteriorVariance
            model = PairwiseProbitModel

            [par1]
            par_type = continuous
            lower_bound = 0 # lower bound
            upper_bound = 1 # upper bound

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            min_asks = 10
            generator = SobolGenerator

            [opt_strat]
            min_asks = 20
            generator = OptimizeAcqfGenerator

            [PairwiseProbitModel]
            mean_covar_factory = default_mean_covar_factory

            [PairwiseMCPosteriorVariance]
            objective = ProbitObjective

            [OptimizeAcqfGenerator]
            restarts = 10
            samps = 1000

            [SobolGenerator]
            n_points = 20
            """
        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)

        self.assertTrue(isinstance(strat.strat_list[0].generator, SobolGenerator))
        self.assertTrue(isinstance(strat.strat_list[1].model, PairwiseProbitModel))
        self.assertTrue(
            strat.strat_list[1].generator.acqf is PairwiseMCPosteriorVariance
        )
        # because ProbitObjective() is an object, test keys then vals
        self.assertTrue(
            set(strat.strat_list[1].generator.acqf_kwargs.keys()) == {"objective"}
        )
        self.assertTrue(
            isinstance(
                strat.strat_list[1].generator.acqf_kwargs["objective"],
                ProbitObjective,
            )
        )

        self.assertTrue(strat.strat_list[1].generator.restarts == 10)
        self.assertTrue(strat.strat_list[1].generator.samps == 1000)
        self.assertTrue(strat.strat_list[0].min_asks == 10)
        self.assertTrue(strat.strat_list[0].stimuli_per_trial == 2)
        self.assertTrue(strat.strat_list[0].outcome_types == ["binary"])
        self.assertTrue(strat.strat_list[1].min_asks == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(
            torch.all(strat.strat_list[1].generator.lb == torch.Tensor([0, 0]))
        )
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(
            torch.all(strat.strat_list[1].generator.ub == torch.Tensor([1, 1]))
        )

    def test_pairwise_probit_config_file(self):
        config_file = "../configs/pairwise_al_example.ini"
        config_file = os.path.join(os.path.dirname(__file__), config_file)

        config = Config()
        config.update(config_fnames=[config_file])
        strat = SequentialStrategy.from_config(config)

        self.assertTrue(isinstance(strat.strat_list[0].generator, SobolGenerator))
        self.assertTrue(strat.strat_list[0].model is None)

        self.assertTrue(isinstance(strat.strat_list[1].model, PairwiseProbitModel))
        self.assertTrue(
            strat.strat_list[1].generator.acqf is PairwiseMCPosteriorVariance
        )
        # because ProbitObjective() is an object, we have to be a bit careful with
        # this test
        self.assertTrue(
            set(strat.strat_list[1].generator.acqf_kwargs.keys()) == {"objective"}
        )
        self.assertTrue(
            isinstance(
                strat.strat_list[1].generator.acqf_kwargs["objective"],
                ProbitObjective,
            )
        )

        self.assertTrue(strat.strat_list[1].generator.restarts == 10)
        self.assertTrue(strat.strat_list[1].generator.samps == 1000)
        self.assertTrue(strat.strat_list[0].min_asks == 10)
        self.assertTrue(strat.strat_list[0].stimuli_per_trial == 2)
        self.assertTrue(strat.strat_list[0].outcome_types == ["binary"])
        self.assertTrue(strat.strat_list[1].min_asks == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(
            torch.all(strat.strat_list[1].generator.lb == torch.Tensor([0, 0]))
        )
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(
            torch.all(strat.strat_list[1].generator.ub == torch.Tensor([1, 1]))
        )

    def test_pairwise_al_config_file(self):
        # random datebase path name without dashes
        database_path = "./{}.db".format(str(uuid.uuid4().hex))
        server = AEPsychServer(database_path=database_path)

        config_file = "../configs/pairwise_al_example.ini"
        config_file = os.path.join(os.path.dirname(__file__), config_file)
        configure(server, config_fnames=[config_file])
        strat = server.strat

        self.assertTrue(isinstance(strat.strat_list[0].generator, SobolGenerator))
        self.assertTrue(strat.strat_list[0].model is None)

        self.assertTrue(
            isinstance(strat.strat_list[1].generator, OptimizeAcqfGenerator)
        )
        self.assertTrue(isinstance(strat.strat_list[1].model, PairwiseProbitModel))
        self.assertTrue(
            strat.strat_list[1].generator.acqf is PairwiseMCPosteriorVariance
        )
        self.assertTrue(
            set(strat.strat_list[1].generator.acqf_kwargs.keys()) == {"objective"}
        )
        self.assertTrue(
            isinstance(
                strat.strat_list[1].generator.acqf_kwargs["objective"],
                ProbitObjective,
            )
        )

        self.assertTrue(strat.strat_list[1].generator.restarts == 10)
        self.assertTrue(strat.strat_list[1].generator.samps == 1000)
        self.assertTrue(strat.strat_list[0].min_asks == 10)
        self.assertTrue(strat.strat_list[0].stimuli_per_trial == 2)
        self.assertTrue(strat.strat_list[0].outcome_types == ["binary"])
        self.assertTrue(strat.strat_list[1].min_asks == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(
            torch.all(strat.strat_list[1].generator.lb == torch.Tensor([0, 0]))
        )
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(
            torch.all(strat.strat_list[1].generator.ub == torch.Tensor([1, 1]))
        )
        # cleanup the db
        if server.db is not None:
            server.db.delete_db()

    def test_pairwise_opt_config(self):
        # random datebase path name without dashes
        database_path = "./{}.db".format(str(uuid.uuid4().hex))
        server = AEPsychServer(database_path=database_path)

        config_file = "../configs/pairwise_opt_example.ini"
        config_file = os.path.join(os.path.dirname(__file__), config_file)

        configure(server, config_fnames=[config_file])
        strat = server.strat

        self.assertTrue(isinstance(strat.strat_list[0].generator, SobolGenerator))
        self.assertTrue(strat.strat_list[0].model is None)

        self.assertTrue(isinstance(strat.strat_list[1].model, PairwiseProbitModel))
        self.assertTrue(
            strat.strat_list[1].generator.acqf is qLogNoisyExpectedImprovement
        )
        self.assertTrue(
            set(strat.strat_list[1].generator.acqf_kwargs.keys()) == {"objective"}
        )
        self.assertTrue(
            isinstance(
                strat.strat_list[1].generator.acqf_kwargs["objective"],
                ProbitObjective,
            )
        )

        self.assertTrue(strat.strat_list[1].generator.restarts == 10)
        self.assertTrue(strat.strat_list[1].generator.samps == 1000)
        self.assertTrue(strat.strat_list[0].min_asks == 10)
        self.assertTrue(strat.strat_list[0].stimuli_per_trial == 2)
        self.assertTrue(strat.strat_list[0].outcome_types == ["binary"])
        self.assertTrue(strat.strat_list[1].min_asks == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(
            torch.all(strat.strat_list[1].generator.lb == torch.Tensor([0, 0]))
        )
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(
            torch.all(strat.strat_list[1].generator.ub == torch.Tensor([1, 1]))
        )
        # cleanup the db
        if server.db is not None:
            server.db.delete_db()

    def test_jsonify(self):
        sample_configstr = """
            [common]
            outcome_type = pairwise_probit
            parnames = [par1, par2]
            strategy_names = [init_strat, opt_strat]
            acqf = PairwiseMCPosteriorVariance
            model = PairwiseProbitModel

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            min_asks = 10
            generator = PairwiseSobolGenerator

            [opt_strat]
            min_asks = 20
            generator = PairwiseOptimizeAcqfGenerator

            [PairwiseProbitModel]
            mean_covar_factory = default_mean_covar_factory

            [PairwiseMCPosteriorVariance]
            objective = ProbitObjective

            [PairwiseOptimizeAcqfGenerator]
            restarts = 10
            samps = 1000

            [PairwiseSobolGenerator]
            n_points = 20
            """
        request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": sample_configstr},
        }
        # Generate a configuration object.
        temporaryconfig = Config(**request["message"])
        configedjson = temporaryconfig.jsonifyAll()
        referencejsonstr = """{
            "common": {
                "outcome_type": "pairwise_probit",
                "parnames": "[par1, par2]",
                "strategy_names": "[init_strat, opt_strat]",
                "acqf": "PairwiseMCPosteriorVariance",
                "model": "PairwiseProbitModel",
                "lb": "[0, 0]",
                "ub": "[1, 1]"
            },
            "par1": {
                "par_type": "continuous",
                "lower_bound": "0",
                "upper_bound": "1"
            },
            "par2": {
                "par_type": "continuous",
                "lower_bound": "0",
                "upper_bound": "1"
            },
            "init_strat": {
                "min_asks": "10",
                "generator": "PairwiseSobolGenerator"
            },
            "opt_strat": {
                "min_asks": "20",
                "generator": "PairwiseOptimizeAcqfGenerator"
            },
            "PairwiseProbitModel": {
                "mean_covar_factory": "default_mean_covar_factory"
            },
            "PairwiseMCPosteriorVariance": {
                "objective": "ProbitObjective"
            },
            "PairwiseOptimizeAcqfGenerator": {
                "restarts": "10",
                "samps": "1000"
            },
            "PairwiseSobolGenerator": {
                "n_points": "20"
            }
        } """
        # Rather than comparing strings, we should convert to json and then convert back to test equal dicts
        testconfig = json.loads(configedjson)
        testsample = json.loads(referencejsonstr)
        # most depth is option within section
        self.assertEqual(testconfig, testsample)

    def test_stimuli_compatibility(self):
        config_str1 = """
            [common]
            stimuli_per_trial = 1
            outcome_types = [binary]
            parnames = [par1, par2]
            strategy_names = [init_strat]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = SobolGenerator
            model = GPClassificationModel
            """
        config1 = Config()
        config1.update(config_str=config_str1)

        config_str2 = """
            [common]
            stimuli_per_trial = 1
            outcome_types = [binary]
            parnames = [par1, par2]
            strategy_names = [init_strat]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = SobolGenerator
            model = GPClassificationModel
            """
        config2 = Config()
        config2.update(config_str=config_str2)

        config_str3 = """
            [common]
            stimuli_per_trial = 1
            outcome_types = [binary]
            parnames = [par1, par2]
            strategy_names = [init_strat]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = SobolGenerator
            model = PairwiseProbitModel
            """
        config3 = Config()
        config3.update(config_str=config_str3)

        # this should work
        SequentialStrategy.from_config(config1)

        # this should fail
        with self.assertRaises(AssertionError):
            SequentialStrategy.from_config(config3)

        # this should fail too
        with self.assertRaises(AssertionError):
            SequentialStrategy.from_config(config3)

    def test_outcome_compatibility(self):
        config_str1 = """
            [common]
            stimuli_per_trial = 1
            outcome_types = [binary]
            parnames = [par1, par2]
            strategy_names = [init_strat]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = SobolGenerator
            model = GPClassificationModel
            """
        config1 = Config()
        config1.update(config_str=config_str1)

        config_str2 = """
            [common]
            stimuli_per_trial = 1
            outcome_types = [continuous]
            parnames = [par1, par2]
            strategy_names = [init_strat]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = SobolGenerator
            model = GPClassificationModel
            """
        config2 = Config()
        config2.update(config_str=config_str2)

        config_str3 = """
            [common]
            stimuli_per_trial = 1
            outcome_types = [binary]
            parnames = [par1, par2]
            strategy_names = [init_strat]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = SobolGenerator
            model = GPRegressionModel
            """
        config3 = Config()
        config3.update(config_str=config_str3)

        # this should work
        SequentialStrategy.from_config(config1)

        # this should fail
        with self.assertRaises(AssertionError):
            SequentialStrategy.from_config(config3)

        # this should fail too
        with self.assertRaises(AssertionError):
            SequentialStrategy.from_config(config3)

    def test_strat_names(self):
        good_str = """
            [common]
            stimuli_per_trial = 1
            outcome_types = [binary]
            parnames = [par1, par2]
            strategy_names = [init strat, opt_strat]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init strat]
            generator = SobolGenerator
            model = GPClassificationModel

            [opt_strat]
            generator = OptimizeAcqfGenerator
            model = GPClassificationModel
            """

        bad_str = """
            [common]
            stimuli_per_trial = 1
            outcome_types = [binary]
            parnames = [par1, par2]
            strategy_names = [init_strat, init_strat]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = SobolGenerator
            model = GPClassificationModel
            """

        good_config = Config(config_str=good_str)
        bad_config = Config(config_str=bad_str)

        # this should work
        strat = SequentialStrategy.from_config(good_config)
        self.assertTrue(strat.strat_list[0].name == "init strat")

        # this should fail
        with self.assertRaises(AssertionError):
            SequentialStrategy.from_config(bad_config)

    def test_semip_config(self):
        config_str = """
            [common]
            stimuli_per_trial = 1
            outcome_types = [binary]
            parnames = [par1, par2]
            strategy_names = [init_strat, opt_strat]
            acqf = MCLevelSetEstimation
            model = HadamardSemiPModel

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            min_asks = 10
            generator = SobolGenerator
            refit_every = 10

            [opt_strat]
            min_asks = 20
            generator = OptimizeAcqfGenerator

            [HadamardSemiPModel]
            stim_dim = 1
            inducing_size = 10
            inducing_point_method = SobolAllocator
            likelihood = BernoulliObjectiveLikelihood

            [BernoulliObjectiveLikelihood]
            objective = FloorGumbelObjective

            [FloorGumbelObjective]
            floor = 0.123

            [OptimizeAcqfGenerator]
            restarts = 10
            samps = 1000

            """

        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)
        opt_strat = strat.strat_list[1]
        model = opt_strat.model
        generator = opt_strat.generator

        self.assertTrue(isinstance(model, HadamardSemiPModel))
        self.assertTrue(torch.all(generator.lb == torch.Tensor([0, 0])))
        self.assertTrue(torch.all(generator.ub == torch.Tensor([1, 1])))
        self.assertTrue(model.inducing_size == 10)
        self.assertTrue(model.stim_dim == 1)

        # Verify the allocator and bounds
        self.assertTrue(isinstance(model.inducing_point_method, SobolAllocator))
        expected_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        self.assertTrue(
            torch.equal(model.inducing_point_method.bounds, expected_bounds)
        )

        self.assertTrue(isinstance(model.likelihood, BernoulliObjectiveLikelihood))
        self.assertTrue(isinstance(model.likelihood.objective, FloorGumbelObjective))

    def test_derived_bounds(self):
        config_str = """
            [common]
            parnames = [par1, par2]
            stimuli_per_trial = 1
            outcome_types = [binary]
            target = 0.75
            strategy_names = [init_strat, opt_strat]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [par2]
            par_type = continuous
            lower_bound = -10
            upper_bound = 10
            normalize_scale = False

            [init_strat]
            min_total_tells = 10
            generator = SobolGenerator

            [opt_strat]
            min_total_tells = 20
            refit_every = 5
            generator = OptimizeAcqfGenerator
            acqf = MCLevelSetEstimation
            model = GPClassificationModel
        """

        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)
        opt_strat = strat.strat_list[1]

        self.assertTrue(torch.all(opt_strat.lb == torch.Tensor([0, -10])))
        self.assertTrue(torch.all(opt_strat.ub == torch.Tensor([1, 10])))

    def test_ignore_common_bounds(self):
        config_str = """
            [common]
            parnames = [par1, par2]
            lb = [0, 0]
            ub = [1, 1]
            stimuli_per_trial = 1
            outcome_types = [binary]
            target = 0.75
            strategy_names = [init_strat, opt_strat]

            [par1]
            par_type = continuous
            lower_bound = 1
            upper_bound = 100

            [par2]
            par_type = continuous
            lower_bound = -5
            upper_bound = 1
            normalize_scale = False

            [init_strat]
            min_total_tells = 10
            generator = SobolGenerator

            [opt_strat]
            min_total_tells = 20
            refit_every = 5
            generator = OptimizeAcqfGenerator
            acqf = MCLevelSetEstimation
            model = GPClassificationModel
        """

        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)
        opt_strat = strat.strat_list[1]

        self.assertTrue(torch.all(opt_strat.lb == torch.Tensor([0, -5])))
        self.assertTrue(torch.all(opt_strat.ub == torch.Tensor([1, 1])))

    def test_common_fallback_bounds(self):
        config_str = """
            [common]
            parnames = [par1, par2]
            lb = [0, 0]
            ub = [1, 100]
            stimuli_per_trial = 1
            outcome_types = [binary]
            target = 0.75
            strategy_names = [init_strat, opt_strat]

            [par1]
            par_type = continuous
            lower_bound = 1
            upper_bound = 100

            [par2]
            par_type = continuous
            # lower_bound = -5
            # upper_bound = 1
            normalize_scale = False

            [init_strat]
            min_total_tells = 10
            generator = SobolGenerator

            [opt_strat]
            min_total_tells = 20
            refit_every = 5
            generator = OptimizeAcqfGenerator
            acqf = MCLevelSetEstimation
            model = GPClassificationModel
        """

        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)
        opt_strat = strat.strat_list[1]

        self.assertTrue(torch.all(opt_strat.lb == torch.Tensor([0, 0])))
        self.assertTrue(torch.all(opt_strat.ub == torch.Tensor([1, 100])))

    def test_parameter_setting_block_validation(self):
        config_str = """
            [common]
            parnames = [par1, par2]
        """
        config = Config()

        with self.assertRaises(ValueError):
            config.update(config_str=config_str)

    def test_invalid_parameter_type(self):
        config_str = """
            [common]
            parnames = [par1]

            [par1]
            par_type = invalid_type
        """
        config = Config()
        with self.assertRaises(ParameterConfigError):
            config.update(config_str=config_str)

    def test_continuous_parameter_lb_validation(self):
        config_str = """
            [common]
            parnames = [par1, par2]

            [par1]
            par_type = continuous
            lower_bound = 1
            upper_bound = 100

            [par2]
            par_type = continuous
            upper_bound = 1
        """
        config = Config()
        with self.assertRaises(ValueError):
            config.update(config_str=config_str)

    def test_continuous_parameter_ub_validation(self):
        config_str = """
            [common]
            parnames = [par1, par2]

            [par1]
            par_type = continuous
            lower_bound = 1
            upper_bound = 100

            [par2]
            par_type = continuous
            lower_bound = 0
        """
        config = Config()
        with self.assertRaises(ValueError):
            config.update(config_str=config_str)

    def test_clone_transform_options(self):
        points = [[0.25, 50], [0.9, 99]]
        window = [0.05, 10]
        config_str = f"""
                [common]
                parnames = [contPar, logPar]
                stimuli_per_trial=1
                outcome_types=[binary]
                target=0.75
                strategy_names = [init_strat]

                [contPar]
                par_type = continuous
                lower_bound = 0
                upper_bound = 1

                [logPar]
                par_type = continuous
                lower_bound = 10
                upper_bound = 100
                log_scale = True

                [init_strat]
                min_total_tells = 10
                generator = SampleAroundPointsGenerator

                [SampleAroundPointsGenerator]
                points = {points}
                window = {window}
            """
        config = Config()
        config.update(config_str=config_str)

        config_clone = transform_options(config)

        self.assertTrue(id(config) != id(config_clone))

        lb = config.gettensor("common", "lb")
        ub = config.gettensor("common", "ub")
        config_points = config.gettensor("SampleAroundPointsGenerator", "points")
        config_window = config.gettensor("SampleAroundPointsGenerator", "window")
        xformed_lb = config_clone.gettensor("common", "lb")
        xformed_ub = config_clone.gettensor("common", "ub")
        xformed_points = config_clone.gettensor("SampleAroundPointsGenerator", "points")
        xformed_window = config_clone.gettensor("SampleAroundPointsGenerator", "window")

        self.assertFalse(torch.all(lb == xformed_lb))
        self.assertFalse(torch.all(ub == xformed_ub))
        self.assertFalse(torch.all(config_points == xformed_points))
        self.assertFalse(torch.all(config_window == xformed_window))

        self.assertTrue(torch.allclose(xformed_lb, torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.allclose(xformed_ub, torch.tensor([1.0, 1.0])))

        transforms = ParameterTransforms.from_config(config)
        reversed_points = transforms.untransform(xformed_points)

        self.assertTrue(torch.allclose(reversed_points, torch.tensor(points)))

    def test_build_transform(self):
        config_str = """
            [common]
            parnames = [signal1, signal2]

            [signal1]
            par_type = continuous
            lower_bound = 1
            upper_bound = 100
            log_scale = false

            [signal2]
            par_type = continuous
            lower_bound = 1
            upper_bound = 100
            log_scale = true
        """
        config = Config()
        config.update(config_str=config_str)

        transforms = ParameterTransforms.from_config(config)

        self.assertTrue(len(transforms.values()) == 3)

        tf = list(transforms.items())[0]
        expected_names = [
            "signal1_NormalizeScale",
            "signal2_Log10Plus",
            "signal2_NormalizeScale",
        ]
        expected_transforms = [NormalizeScale, Log10Plus, NormalizeScale]
        for tf, name, transform in zip(
            transforms.items(), expected_names, expected_transforms
        ):
            self.assertTrue(tf[0] == name)
            self.assertTrue(isinstance(tf[1], transform))

    def test_optimizer_options_smoketest(self):
        config_str = """
            [common]
            parnames = [signal1]
            outcome_types = [binary]
            stimuli_per_trial = 1
            strategy_names = [opt_strat]

            [signal1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [opt_strat]
            model = GPClassificationModel
            generator = SobolGenerator
            min_asks = 1

            [GPClassificationModel]
            maxcor = 1
            maxfun = 0
            maxls = 3
            gtol = 4
            ftol = 5
            maxiter = 6
            max_fit_time = 100
        """
        config = Config()
        config.update(config_str=config_str)

        strat = Strategy.from_config(config, "opt_strat")

        strat.add_data(torch.tensor([0.80]), torch.tensor([1]))
        strat.fit()

        options = strat.model.optimizer_options["options"]
        self.assertTrue(options["maxcor"] == 1)
        self.assertTrue(options["ftol"] == 5.0)
        self.assertTrue(options["gtol"] == 4.0)
        self.assertTrue(options["maxiter"] == 6)
        self.assertTrue(options["maxls"] == 3)

        self.assertTrue(
            options["maxfun"] != 0, "maxfun should be overridden by max_fit_time"
        )


if __name__ == "__main__":
    unittest.main()
