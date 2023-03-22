#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import uuid
from ax import Data

import numpy as np
import torch
from aepsych_client import AEPsychClient
from ax.service.utils.report_utils import exp_to_df
from scipy.stats import norm
from aepsych.benchmark.test_functions import novel_detection_testfun
from torch.distributions import Categorical

from aepsych.config import Config
from aepsych.server import AEPsychServer
from parameterized import parameterized_class


@parameterized_class(
    ("config_file", "should_ignore"),
    [
        ("../configs/ax_example.ini", False),
        ("../configs/ax_ordinal_exploration_example.ini", True),
    ],
)
class AxIntegrationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Skip test if it should be ignored
        if cls.should_ignore:
            return
        # Simulate participant responses; just returns the sum of the flat parameters
        def simulate_response(trial_params):
            pars = [
                trial_params[par][0]
                for par in trial_params
                if type(trial_params[par][0]) == float
            ]
            response = np.array(pars).sum()
            return response

        # Fix random seeds
        np.random.seed(0)
        torch.manual_seed(0)

        # Create a server object configured to run a 2d threshold experiment
        database_path = "./{}.db".format(str(uuid.uuid4().hex))
        cls.client = AEPsychClient(server=AEPsychServer(database_path=database_path))
        cls.config_file = os.path.join(os.path.dirname(__file__), cls.config_file)
        cls.client.configure(cls.config_file)

        while not cls.client.server.strat.finished:
            # Ask the server what the next parameter values to test should be.
            trial_params = cls.client.ask()

            # Simulate a participant response.
            outcome = simulate_response(trial_params["config"])

            # Tell the server what happened so that it can update its model.
            cls.client.tell(trial_params["config"], outcome)

        # Add an extra tell to make sure manual tells and duplicate params
        cls.client.tell(trial_params["config"], outcome)

        cls.df = exp_to_df(cls.client.server.strat.experiment)

        cls.config = Config(config_fnames=[cls.config_file])

    def tearDown(self):
        # Skip test if it should be ignored
        if self.should_ignore:
            return
        if self.client.server.db is not None:
            self.client.server.db.delete_db()

    def test_bounds(self):
        # Skip test if it should be ignored
        if self.should_ignore:
            return
        lb = self.config.getlist("common", "lb", element_type=float)
        ub = self.config.getlist("common", "ub", element_type=float)
        par4choices = self.config.getlist("par4", "choices", element_type=str)
        par5choices = self.config.getlist("par5", "choices", element_type=str)
        par6value = self.config.getfloat("par6", "value")
        par7value = self.config.get("par7", "value")

        self.assertTrue((self.df["par1"] >= lb[0]).all())
        self.assertTrue((self.df["par1"] <= ub[0]).all())

        self.assertTrue((self.df["par2"] >= lb[1]).all())
        self.assertTrue((self.df["par2"] <= ub[1]).all())

        self.assertTrue((self.df["par3"] >= lb[2]).all())
        self.assertTrue((self.df["par3"] <= ub[2]).all())

        self.assertTrue(self.df["par4"].isin(par4choices).all())

        self.assertTrue(self.df["par5"].isin(par5choices).all())

        self.assertTrue((self.df["par6"] == par6value).all())

        self.assertTrue((self.df["par7"] == par7value).all())

    def test_constraints(self):
        # Skip test if it should be ignored
        if self.should_ignore:
            return
        constraints = self.config.getlist("common", "par_constraints", element_type=str)
        for constraint in constraints:
            self.assertEqual(len(self.df.query(constraint)), len(self.df))

        self.assertEqual(self.df["par3"].dtype, "int64")

    def test_n_trials(self):
        # Skip test if it should be ignored
        if self.should_ignore:
            return
        n_tells = (self.df["trial_status"] == "COMPLETED").sum()
        correct_n_tells = self.config.getint("opt_strat", "min_total_tells") + 1

        self.assertEqual(n_tells, correct_n_tells)

    def test_generation_method(self):
        # Skip test if it should be ignored
        if self.should_ignore:
            return
        n_sobol = (self.df["generation_method"] == "Sobol").sum()
        n_opt = (self.df["generation_method"] == "BoTorch").sum()

        correct_n_sobol = self.config.getint("init_strat", "min_total_tells")
        correct_n_opt = (
            self.config.getint("opt_strat", "min_total_tells") - correct_n_sobol
        )

        self.assertEqual(n_sobol, correct_n_sobol)
        self.assertEqual(n_opt, correct_n_opt)
