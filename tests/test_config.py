import os
import unittest

import torch
from aepsych.acquisition.lse import LevelSetEstimation
from aepsych.acquisition.monotonic_rejection import MonotonicMCLSE
from aepsych.acquisition.objective import ProbitObjective
from aepsych.config import Config
from aepsych.modelbridge import (
    MonotonicSingleProbitModelbridge,
    SingleProbitModelbridge,
)
from aepsych.strategy import (
    ModelWrapperStrategy,
    SequentialStrategy,
    SobolStrategy,
)


class ConfigTestCase(unittest.TestCase):
    def test_single_probit_config(self):
        config_str = """
        [common]
        lb = [0, 0]
        ub = [1, 1]
        outcome_type = single_probit
        parnames = [par1, par2]

        [experiment]
        acqf = LevelSetEstimation
        modelbridge_cls = SingleProbitModelbridge
        init_strat_cls = SobolStrategy
        opt_strat_cls = ModelWrapperStrategy

        [LevelSetEstimation]
        beta = 3.98

        [GPClassificationModel]
        inducing_size = 10
        mean_covar_factory = default_mean_covar_factory

        [SingleProbitModelbridge]
        restarts = 10
        samps = 1000

        [SobolStrategy]
        n_trials = 10

        [ModelWrapperStrategy]
        n_trials = 20
        """
        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)

        self.assertTrue(isinstance(strat.strat_list[0], SobolStrategy))
        self.assertTrue(isinstance(strat.strat_list[1], ModelWrapperStrategy))
        self.assertTrue(
            isinstance(strat.strat_list[1].modelbridge, SingleProbitModelbridge)
        )
        self.assertTrue(strat.strat_list[1].modelbridge.acqf is LevelSetEstimation)
        # since ProbitObjective() is turned into an obj, we check for keys and then vals
        self.assertTrue(
            set(strat.strat_list[1].modelbridge.extra_acqf_args.keys())
            == {"beta", "target", "objective"}
        )
        self.assertTrue(
            strat.strat_list[1].modelbridge.extra_acqf_args["target"] == 0.75
        )
        self.assertTrue(strat.strat_list[1].modelbridge.extra_acqf_args["beta"] == 3.98)
        self.assertTrue(
            isinstance(
                strat.strat_list[1].modelbridge.extra_acqf_args["objective"],
                ProbitObjective,
            )
        )

        self.assertTrue(strat.strat_list[1].modelbridge.restarts == 10)
        self.assertTrue(strat.strat_list[1].modelbridge.samps == 1000)
        self.assertTrue(strat.strat_list[0].n_trials == 10)
        self.assertTrue(strat.strat_list[0].outcome_type == "single_probit")
        self.assertTrue(strat.strat_list[1].n_trials == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(
            torch.all(strat.strat_list[1].modelbridge.lb == torch.Tensor([0, 0]))
        )
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(
            torch.all(strat.strat_list[1].modelbridge.ub == torch.Tensor([1, 1]))
        )

    def test_monotonic_single_probit_config_file(self):
        config_file = "../configs/single_lse_example.ini"
        config_file = os.path.join(os.path.dirname(__file__), config_file)

        config = Config()
        config.update(config_fnames=[config_file])
        strat = SequentialStrategy.from_config(config)

        self.assertTrue(isinstance(strat.strat_list[0], SobolStrategy))
        self.assertTrue(isinstance(strat.strat_list[1], ModelWrapperStrategy))
        self.assertTrue(
            isinstance(
                strat.strat_list[1].modelbridge, MonotonicSingleProbitModelbridge
            )
        )
        self.assertTrue(strat.strat_list[1].modelbridge.acqf is MonotonicMCLSE)
        self.assertTrue(
            strat.strat_list[1].modelbridge.extra_acqf_args
            == {"beta": 3.98, "target": 0.75}
        )
        self.assertTrue(strat.strat_list[1].modelbridge.samps == 1000)
        self.assertTrue(strat.strat_list[0].n_trials == 10)
        self.assertTrue(strat.strat_list[0].outcome_type == "single_probit")
        self.assertTrue(strat.strat_list[1].n_trials == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(
            torch.all(strat.strat_list[1].modelbridge.lb == torch.Tensor([0, 0]))
        )
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(
            torch.all(strat.strat_list[1].modelbridge.ub == torch.Tensor([1, 1]))
        )

    def test_name_conflict_warns(self):
        class DummyMod:
            pass

        Config.register_object(DummyMod)

        with self.assertWarns(Warning):
            Config.register_object(DummyMod)
