#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from ..test_server import BaseServerTestCase


# Smoke test to make sure nothing breaks. This should really be combined with
# the individual query tests
class QueryHandlerTestCase(BaseServerTestCase):
    def test_strat_query(self):
        # Annoying and complex model and output shapes
        config_str = """
            [common]
            stimuli_per_trial=2
            outcome_types=[binary]
            parnames = [par1, par2, par3]
            strategy_names = [opt_strat]
            acqf = PairwiseMCPosteriorVariance

            [par1]
            par_type = continuous
            lower_bound = -1
            upper_bound = 1

            [par2]
            par_type = continuous
            lower_bound = -1
            upper_bound = 1

            [par3]
            par_type = continuous
            lower_bound = 10
            upper_bound = 100

            [opt_strat]
            min_asks = 1
            model = PairwiseProbitModel
            generator = OptimizeAcqfGenerator

            [PairwiseProbitModel]
            mean_covar_factory = default_mean_covar_factory

            [PairwiseMCPosteriorVariance]
            objective = ProbitObjective

            [OptimizeAcqfGenerator]
            restarts = 10
            samps = 1000
            """
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": config_str},
        }
        ask_request = {"type": "ask", "message": ""}
        tell_request = {
            "type": "tell",
            "message": [
                {
                    "config": {
                        "par1": [0.5, 0.5],
                        "par2": [-0.5, -0.5],
                        "par3": [40, 50],
                    },
                    "outcome": 1,
                },
                {
                    "config": {
                        "par1": [0.0, 0.75],
                        "par2": [0.0, -1],
                        "par3": [11, 99],
                    },
                    "outcome": 0,
                },
                {
                    "config": {"par1": [1, -1], "par2": [0, 0.0], "par3": [40, 12]},
                    "outcome": 0,
                },
            ],
        }

        self.s.handle_request(setup_request)
        while not self.s.strat.finished:
            self.s.handle_request(tell_request)
            self.s.handle_request(ask_request)

        query_max_req = {
            "type": "query",
            "message": {
                "query_type": "max",
            },
        }
        query_min_req = {
            "type": "query",
            "message": {
                "query_type": "min",
            },
        }
        query_pred_req = {
            "type": "query",
            "message": {
                "query_type": "prediction",
                "x": {"par1": [0.0], "par2": [-0.5], "par3": [45]},
            },
        }
        query_inv_req = {
            "type": "query",
            "message": {
                "query_type": "inverse",
                "y": 5.0,
            },
        }

        query_max_const = {
            "type": "query",
            "message": {"query_type": "max", "constraints": {1: 0}},
        }
        query_min_const = {
            "type": "query",
            "message": {"query_type": "min", "constraints": {0: 0.25}},
        }
        query_inv_const = {
            "type": "query",
            "message": {"query_type": "inverse", "y": 5.0, "constraints": {2: 20}},
        }

        response = self.s.handle_request(query_min_req)
        self.assertTrue(len(response["x"]["par1"]) == 1)
        self.assertTrue(len(response["x"]["par2"]) == 1)

        response = self.s.handle_request(query_max_req)
        self.assertTrue(len(response["x"]["par1"]) == 1)
        self.assertTrue(len(response["x"]["par2"]) == 1)

        response = self.s.handle_request(query_inv_req)
        self.assertTrue(len(response["x"]["par1"]) == 1)
        self.assertTrue(len(response["x"]["par2"]) == 1)

        response = self.s.handle_request(query_pred_req)
        self.assertTrue(len(response["x"]["par1"]) == 1)
        self.assertTrue(len(response["x"]["par2"]) == 1)

        response = self.s.handle_request(query_max_const)
        self.assertTrue(response["x"]["par2"][0] == 0)

        response = self.s.handle_request(query_min_const)
        self.assertTrue(response["x"]["par1"][0] == 0.25)

        response = self.s.handle_request(query_inv_const)
        self.assertTrue(response["x"]["par3"][0] == 20)

    def test_grad_model_smoketest(self):
        # Some models return values with gradients that need to be handled
        config_str = """
            [common]
            lb = [0, 0]
            ub = [1, 1]
            parnames = [x, y]
            stimuli_per_trial = 2
            outcome_types = [binary]
            strategy_names = [init_strat, opt_strat]

            [init_strat]
            min_asks = 2
            generator = SobolGenerator
            min_total_outcome_occurrences = 0

            [opt_strat]
            min_asks = 2
            generator = OptimizeAcqfGenerator
            acqf = qLogNoisyExpectedImprovement
            model = PairwiseProbitModel
            min_total_outcome_occurrences = 0

            [qLogNoisyExpectedImprovement]
            objective = ProbitObjective
        """

        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": config_str},
        }
        ask_request = {"type": "ask", "message": ""}
        tell_request = {
            "type": "tell",
            "message": [
                {"config": {"x": [0.5, 0.5], "y": [0.25, 0.75]}, "outcome": 1},
                {"config": {"x": [0.25, 0.75], "y": [0.5, 0.5]}, "outcome": 0},
                {"config": {"x": [0.2, 0.6], "y": [0.3, 0.9]}, "outcome": 0},
            ],
        }

        self.s.handle_request(setup_request)
        while not self.s.strat.finished:
            self.s.handle_request(ask_request)
            self.s.handle_request(tell_request)

        query_max_req = {
            "type": "query",
            "message": {
                "query_type": "max",
            },
        }
        query_min_req = {
            "type": "query",
            "message": {
                "query_type": "min",
                "probability_space": True,
            },
        }
        query_pred_req = {
            "type": "query",
            "message": {
                "query_type": "prediction",
                "x": {"x": [0.0], "y": [1.0]},
            },
        }
        query_pred_prob = {
            "type": "query",
            "message": {
                "query_type": "prediction",
                "x": {"x": [0.0], "y": [1.0]},
                "probability_space": True,
            },
        }
        query_inv_req = {
            "type": "query",
            "message": {
                "query_type": "inverse",
                "y": 5.0,
            },
        }

        self.s.handle_request(query_min_req)
        self.s.handle_request(query_pred_req)
        self.s.handle_request(query_pred_prob)
        self.s.handle_request(query_max_req)
        self.s.handle_request(query_inv_req)


if __name__ == "__main__":
    unittest.main()
