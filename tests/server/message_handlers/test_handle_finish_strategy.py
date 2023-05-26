#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from ..test_server import BaseServerTestCase, dummy_config


class ResumeTestCase(BaseServerTestCase):
    def test_handle_finish_strategy(self):
        setup_request = {
            "type": "setup",
            "message": {"config_str": dummy_config},
        }

        tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5]}, "outcome": 1},
        }

        ask_request = {"type": "ask", "message": ""}

        strat_name_request = {"type": "info"}
        finish_strat_request = {"type": "finish_strategy"}

        self.s.handle_request(setup_request)
        strat_name = self.s.handle_request(strat_name_request)["current_strat_name"]
        self.assertEqual(strat_name, "init_strat")

        # model-based strategies require data
        self.s.handle_request(tell_request)

        msg = self.s.handle_request(finish_strat_request)
        self.assertEqual(msg, "finished strategy init_strat")

        # need to gen another trial to move to next strategy
        self.s.handle_request(ask_request)

        strat_name = self.s.handle_request(strat_name_request)["current_strat_name"]
        self.assertEqual(strat_name, "opt_strat")


if __name__ == "__main__":
    unittest.main()
