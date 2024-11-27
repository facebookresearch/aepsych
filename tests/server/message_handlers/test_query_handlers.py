#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from ..test_server import BaseServerTestCase, dummy_config


# Smoke test to make sure nothing breaks. This should really be combined with
# the individual query tests
class QueryHandlerTestCase(BaseServerTestCase):
    def test_strat_query(self):
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": dummy_config},
        }
        ask_request = {"type": "ask", "message": ""}
        tell_request = {
            "type": "tell",
            "message": [
                {"config": {"x": [0.5]}, "outcome": 1},
                {"config": {"x": [0.0]}, "outcome": 0},
                {"config": {"x": [1]}, "outcome": 0},
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
            },
        }
        query_pred_req = {
            "type": "query",
            "message": {
                "query_type": "prediction",
                "x": {"x": [0.0]},
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
        self.s.handle_request(query_max_req)
        self.s.handle_request(query_inv_req)

    def test_grad_model(self):
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
            },
        }
        query_pred_req = {
            "type": "query",
            "message": {
                "query_type": "prediction",
                "x": {"x": [0.0], "y": [1.0]},
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
        self.s.handle_request(query_max_req)
        self.s.handle_request(query_inv_req)


if __name__ == "__main__":
    unittest.main()
