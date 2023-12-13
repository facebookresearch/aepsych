#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import uuid

import torch

from aepsych.server import AEPsychServer
from aepsych_client import AEPsychClient
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.modelbridge import Models
from botorch.test_functions.multi_objective import BraninCurrin

branin_currin = BraninCurrin(negate=True).to(
    dtype=torch.double,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


def evaluate(parameters):
    evaluation = branin_currin(
        torch.tensor([parameters.get("x1"), parameters.get("x2")])
    )
    # In our case, standard error is 0, since we are computing a synthetic function.
    # Set standard error to None if the noise level is unknown.
    return {"out1": evaluation[0].item(), "out2": evaluation[1].item()}


class MultiOutcomeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a server object configured to run a 2d threshold experiment
        database_path = "./{}.db".format(str(uuid.uuid4().hex))
        cls.client = AEPsychClient(server=AEPsychServer(database_path=database_path))
        config_file = "../configs/multi_outcome_example.ini"
        config_file = os.path.join(os.path.dirname(__file__), config_file)
        cls.client.configure(config_file)
        cls.gs = cls.client.server.strat.ax_client.generation_strategy
        cls.experiment = cls.client.server.strat.ax_client.experiment

    def test_generation_strategy(self):
        self.assertEqual(len(self.gs._steps), 2 + 1)
        self.assertEqual(self.gs._steps[0].model, Models.SOBOL)
        self.assertEqual(self.gs._steps[1].model, Models.BOTORCH_MODULAR)

    def test_experiment(self):
        self.assertEqual(len(self.experiment.metrics), 2)
        self.assertIn("out1", self.experiment.metrics)
        self.assertIn("out2", self.experiment.metrics)
        self.assertIsInstance(
            self.experiment.optimization_config, MultiObjectiveOptimizationConfig
        )
        (
            threshold1,
            threshold2,
        ) = self.experiment.optimization_config.objective_thresholds
        self.assertEqual(threshold1.bound, -18)
        self.assertEqual(threshold2.bound, -6)
        (
            objective1,
            objective2,
        ) = self.experiment.optimization_config.objective.objectives
        self.assertFalse(objective1.minimize)
        self.assertFalse(objective2.minimize)

    # Smoke test just to make sure server can handle multioutcome messages
    def test_ask_tell(self):
        while not self.client.server.strat.finished:
            trial_params = self.client.ask()
            for trial in trial_params["config"]:
                outcome = evaluate(trial_params["config"][trial])
                self.client.tell_trial_by_index(trial, outcome)


if __name__ == "__main__":
    unittest.main()
