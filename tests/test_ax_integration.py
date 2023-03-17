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


# class AxIntegrationTestCase(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         # Simulate participant responses; just returns the sum of the flat parameters
#         def simulate_response(trial_params):
#             pars = [
#                 trial_params[par][0]
#                 for par in trial_params
#                 if type(trial_params[par][0]) == float
#             ]
#             response = np.array(pars).sum()
#             return response

#         # Fix random seeds
#         np.random.seed(0)
#         torch.manual_seed(0)

#         # Create a server object configured to run a 2d threshold experiment
#         database_path = "./{}.db".format(str(uuid.uuid4().hex))
#         cls.client = AEPsychClient(server=AEPsychServer(database_path=database_path))
#         config_file = "../configs/ax_example.ini"
#         config_file = os.path.join(os.path.dirname(__file__), config_file)
#         cls.client.configure(config_file)

#         while not cls.client.server.strat.finished:
#             # Ask the server what the next parameter values to test should be.
#             trial_params = cls.client.ask()

#             # Simulate a participant response.
#             outcome = simulate_response(trial_params["config"])

#             # Tell the server what happened so that it can update its model.
#             cls.client.tell(trial_params["config"], outcome)

#         # Add an extra tell to make sure manual tells and duplicate params
#         cls.client.tell(trial_params["config"], outcome)

#         cls.df = exp_to_df(cls.client.server.strat.experiment)

#         cls.config = Config(config_fnames=[config_file])

#     def tearDown(self):
#         if self.client.server.db is not None:
#             self.client.server.db.delete_db()

#     def test_bounds(self):
#         lb = self.config.getlist("common", "lb", element_type=float)
#         ub = self.config.getlist("common", "ub", element_type=float)
#         par4choices = self.config.getlist("par4", "choices", element_type=str)
#         par5choices = self.config.getlist("par5", "choices", element_type=str)
#         par6value = self.config.getfloat("par6", "value")
#         par7value = self.config.get("par7", "value")

#         self.assertTrue((self.df["par1"] >= lb[0]).all())
#         self.assertTrue((self.df["par1"] <= ub[0]).all())

#         self.assertTrue((self.df["par2"] >= lb[1]).all())
#         self.assertTrue((self.df["par2"] <= ub[1]).all())

#         self.assertTrue((self.df["par3"] >= lb[2]).all())
#         self.assertTrue((self.df["par3"] <= ub[2]).all())

#         self.assertTrue(self.df["par4"].isin(par4choices).all())

#         self.assertTrue(self.df["par5"].isin(par5choices).all())

#         self.assertTrue((self.df["par6"] == par6value).all())

#         self.assertTrue((self.df["par7"] == par7value).all())

#     def test_constraints(self):
#         constraints = self.config.getlist("common", "par_constraints", element_type=str)
#         for constraint in constraints:
#             self.assertEqual(len(self.df.query(constraint)), len(self.df))

#         self.assertEqual(self.df["par3"].dtype, "int64")

#     def test_n_trials(self):
#         n_tells = (self.df["trial_status"] == "COMPLETED").sum()
#         correct_n_tells = self.config.getint("opt_strat", "min_total_tells") + 1

#         self.assertEqual(n_tells, correct_n_tells)

#     def test_generation_method(self):
#         n_sobol = (self.df["generation_method"] == "Sobol").sum()
#         n_opt = (self.df["generation_method"] == "BoTorch").sum()

#         correct_n_sobol = self.config.getint("init_strat", "min_total_tells")
#         correct_n_opt = (
#             self.config.getint("opt_strat", "min_total_tells") - correct_n_sobol
#         )

#         self.assertEqual(n_sobol, correct_n_sobol)
#         self.assertEqual(n_opt, correct_n_opt)

class AxOrdinalGPTestCase(unittest.TestCase):
    """
    Tests that the Ax integration works with an ordinal GP model.
    """
    @classmethod
    def setUpClass(cls):

        # Fix random seeds
        np.random.seed(0)
        torch.manual_seed(0)

        def make_prob_matrix(fgrid, cutpoints, n_levels):
            """
            Generates the matrix of response probabilities for each choice given function values, cutpoints, 
            and number of levels.
            """
            probs = np.zeros((*fgrid.shape, n_levels))

            probs[..., 0] = norm.cdf(cutpoints[0] - fgrid)

            for i in range(1, n_levels - 1):
                probs[..., i] = norm.cdf(cutpoints[i] - fgrid) - norm.cdf(cutpoints[i - 1] - fgrid)

            probs[..., -1] = 1 - norm.cdf(cutpoints[-1] - fgrid)
            return probs

        # Generate synthetic responses from our model (defined above)
        def simulate_trial(trial_config):
            n_levels = 5
            pars = [
                trial_config[par][0]
                for par in trial_config
                if type(trial_config[par][0]) == float
            ]
            response = np.array(pars)
            x = torch.Tensor(response)
            cutpoints = np.quantile(novel_detection_testfun(x), np.linspace(0.05, 0.95, n_levels - 1))
            p = make_prob_matrix(novel_detection_testfun(x), cutpoints=cutpoints, n_levels=n_levels)
            return Categorical(probs=torch.Tensor(p)).sample(torch.Size([1])).squeeze().item()

        # Create a server object configured to run a 2d threshold experiment
        database_path = "./{}.db".format(str(uuid.uuid4().hex))
        cls.client = AEPsychClient(server=AEPsychServer(database_path=database_path))
        config_file = "../configs/ax_ordinal_exploration_example.ini"
        config_file = os.path.join(os.path.dirname(__file__), config_file)
        cls.client.configure(config_file)

        # run a full synthetic experiment loop
        while not cls.client.server.strat.finished:
            trial_config = cls.client.ask()
            outcome = simulate_trial(trial_config=trial_config['config'])
            cls.client.tell(config=trial_config['config'], outcome=outcome)
            
        cls.client.tell(trial_config["config"], outcome)
        cls.client.finalize()
        cls.df = exp_to_df(cls.client.server.strat.experiment)

        cls.config = Config(config_fnames=[config_file])

    def tearDown(self):
        if self.client.server.db is not None:
            self.client.server.db.delete_db()
    
    def test_bounds(self):
        lb = self.config.getlist("common", "lb", element_type=float)
        ub = self.config.getlist("common", "ub", element_type=float)
        par4choices = self.config.getlist("par4", "choices", element_type=str)
        par5value = self.config.getfloat("par5", "value")

        self.assertTrue((self.df["par1"] >= lb[0]).all())
        self.assertTrue((self.df["par1"] <= ub[0]).all())

        self.assertTrue((self.df["par2"] >= lb[1]).all())
        self.assertTrue((self.df["par2"] <= ub[1]).all())

        self.assertTrue((self.df["par3"] >= lb[2]).all())
        self.assertTrue((self.df["par3"] <= ub[2]).all())

        self.assertTrue(self.df["par4"].isin(par4choices).all())

        self.assertTrue((self.df["par5"] == par5value).all())

    def test_constraints(self):
        constraints = self.config.getlist("common", "par_constraints", element_type=str)
        for constraint in constraints:
            self.assertEqual(len(self.df.query(constraint)), len(self.df))

        self.assertEqual(self.df["par3"].dtype, "int64")

    def test_n_trials(self):
        n_tells = (self.df["trial_status"] == "COMPLETED").sum()
        correct_n_tells = self.config.getint("opt_strat", "min_total_tells") + 1

        self.assertEqual(n_tells, correct_n_tells)

    def test_generation_method(self):
        n_sobol = (self.df["generation_method"] == "Sobol").sum()
        n_opt = (self.df["generation_method"] == "BoTorch").sum()

        correct_n_sobol = self.config.getint("init_strat", "min_total_tells")
        correct_n_opt = (
            self.config.getint("opt_strat", "min_total_tells") - correct_n_sobol
        )

        self.assertEqual(n_sobol, correct_n_sobol)
        self.assertEqual(n_opt, correct_n_opt)



if __name__ == "__main__":
    unittest.main()
