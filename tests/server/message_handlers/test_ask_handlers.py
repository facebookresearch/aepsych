#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from ..test_server import BaseServerTestCase


class AskHandlerTestCase(BaseServerTestCase):
    def test_fixed_ask(self):
        config_str = """
        [common]
        parnames = [par1, par2]
        stimuli_per_trial = 1
        outcome_types = [binary]
        strategy_names = [init_strat, opt_strat]

        [par1]
        par_type = continuous
        lower_bound = 1
        upper_bound = 100

        [par2]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [init_strat]
        generator = SobolGenerator
        min_total_tells = 1

        [opt_strat]
        generator = OptimizeAcqfGenerator
        acqf = MCLevelSetEstimation
        model = GPClassificationModel
        min_total_tells = 2
        """
        setup_request = {
            "type": "setup",
            "message": {"config_str": config_str},
        }
        self.s.handle_request(setup_request)

        fixed1 = 75.0
        fixed2 = 0.75

        # SobolGenerator
        # One fixed
        resp = self.s.handle_request(
            {"type": "ask", "message": {"fixed_pars": {"par1": fixed1}}}
        )
        self.assertTrue(resp["config"]["par1"][0] == fixed1)

        self.s.handle_request(
            {"type": "tell", "message": {"config": resp["config"], "outcome": 1}}
        )

        # Both fixed
        resp = self.s.handle_request(
            {"type": "ask", "message": {"fixed_pars": {"par1": fixed1, "par2": fixed2}}}
        )
        self.assertTrue(resp["config"]["par1"][0] == fixed1)
        self.assertTrue(resp["config"]["par2"][0] == fixed2)

        self.s.handle_request(
            {"type": "tell", "message": {"config": resp["config"], "outcome": 0}}
        )

        # OptimizeAcqfGenerator
        # One fixed
        resp = self.s.handle_request(
            {"type": "ask", "message": {"fixed_pars": {"par1": fixed1}}}
        )
        self.assertTrue(resp["config"]["par1"][0] == fixed1)

        self.s.handle_request(
            {"type": "tell", "message": {"config": resp["config"], "outcome": 1}}
        )

        # All fixed
        resp = self.s.handle_request(
            {
                "type": "ask",
                "message": {"fixed_pars": {"par1": fixed1, "par2": fixed2}},
            }
        )

        self.assertTrue(resp["config"]["par1"][0] == fixed1)
        self.assertTrue(resp["config"]["par2"][0] == fixed2)
