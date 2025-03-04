#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from ..test_server import BaseServerTestCase, dummy_config


class InfoTestCase(BaseServerTestCase):
    def test_info(self):
        setup_request = {
            "type": "setup",
            "message": {"config_str": dummy_config},
        }

        tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5]}, "outcome": 1},
        }

        ask_request = {"type": "ask", "message": ""}

        info_request = {"type": "info"}

        # Handle some messages
        self.s.handle_request(setup_request)
        self.s.handle_request(ask_request)
        self.s.handle_request(tell_request)

        msg = self.s.handle_request(info_request)

        self.assertEqual(msg["db_name"], self.db_name)
        self.assertEqual(msg["exp_id"], "e1")
        self.assertEqual(msg["strat_count"], 1)
        self.assertEqual(msg["current_strat_index"], 0)
        self.assertEqual(msg["current_strat_name"], "init_strat")
        self.assertEqual(msg["current_strat_data_pts"], 1)
        self.assertEqual(msg["current_strat_model"], "model not set")
        self.assertEqual(msg["current_strat_acqf"], "acqf not set")
        self.assertFalse(msg["current_strat_finished"])
        self.assertFalse(msg["current_strat_can_fit"])

        # Tell more data, then ask
        self.s.handle_request(ask_request)
        self.s.handle_request(tell_request)
        self.s.handle_request(ask_request)

        msg = self.s.handle_request(info_request)
        self.assertEqual(msg["current_strat_name"], "opt_strat")
        self.assertEqual(msg["current_strat_data_pts"], 2)
        self.assertEqual(msg["current_strat_model"], "GPClassificationModel")
        self.assertEqual(msg["current_strat_acqf"], "MCPosteriorVariance")
        self.assertFalse(msg["current_strat_finished"])
        self.assertTrue(msg["current_strat_can_fit"])


if __name__ == "__main__":
    unittest.main()
