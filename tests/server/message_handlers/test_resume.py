#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from ..test_server import BaseServerTestCase


class ResumeTestCase(BaseServerTestCase):
    def test_handle_resume(self):
        config_str1 = """
            [common]
            lb = [0]
            ub = [1]
            parnames = [x]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat, opt_strat]

            [metadata]
            experiment_id = e1

            [init_strat]
            min_asks = 3
            generator = SobolGenerator
            min_total_outcome_occurrences = 0

            [opt_strat]
            min_asks = 2
            generator = OptimizeAcqfGenerator
            model = GPClassificationModel
            min_total_outcome_occurrences = 0

            [OptimizeAcqfGenerator]
            acqf = MCPosteriorVariance

            [GPClassificationModel]
            inducing_size = 10
        """

        config_str2 = """
            [common]
            lb = [0]
            ub = [1]
            parnames = [x]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat, opt_strat]

            [metadata]
            experiment_id = e2

            [init_strat]
            min_asks = 3
            generator = SobolGenerator
            min_total_outcome_occurrences = 0

            [opt_strat]
            min_asks = 2
            generator = OptimizeAcqfGenerator
            model = GPClassificationModel
            min_total_outcome_occurrences = 0

            [OptimizeAcqfGenerator]
            acqf = MCPosteriorVariance

            [GPClassificationModel]
            inducing_size = 10
        """

        tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5]}, "outcome": 1},
        }

        ask_request = {"type": "ask", "message": ""}

        strat_name_request = {"type": "info"}
        resume_request = {"type": "resume", "message": {"strat_id": 0}}

        msg = self.s.handle_request(
            {"type": "setup", "message": {"config_str": config_str1}}
        )
        # Give it some data
        self.s.handle_request(tell_request)

        # New experiment
        msg = self.s.handle_request(
            {"type": "setup", "message": {"config_str": config_str2}}
        )
        self.assertEqual(msg["strat_id"], 1)

        self.s.handle_request(ask_request)
        msg = self.s.handle_request(strat_name_request)
        self.assertEqual(msg["exp_id"], "e2")
        self.assertEqual(msg["current_strat_index"], 1)
        self.assertEqual(msg["current_strat_data_pts"], 0)

        # resume the first strategy again
        msg = self.s.handle_request(resume_request)
        self.assertEqual(msg["strat_id"], 0)

        # check state
        msg = self.s.handle_request(strat_name_request)
        self.assertEqual(msg["exp_id"], "e1")
        self.assertEqual(msg["current_strat_index"], 0)
        self.assertEqual(msg["current_strat_data_pts"], 1)
        self.assertEqual(msg["strat_count"], 2)
