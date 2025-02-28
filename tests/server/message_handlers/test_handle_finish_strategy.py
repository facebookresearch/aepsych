#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from ..test_server import BaseServerTestCase, dummy_config


class FinishStrategyTestCase(BaseServerTestCase):
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
        msg = self.s.handle_request(strat_name_request)
        self.assertEqual(msg["current_strat_name"], "init_strat")

        # model-based strategies require data
        self.s.handle_request(tell_request)

        msg = self.s.handle_request(finish_strat_request)
        self.assertEqual(msg["finished_strategy"], "init_strat")
        self.assertEqual(msg["finished_strat_idx"], 0)

        # need to gen another trial to move to next strategy
        self.s.handle_request(ask_request)

        msg = self.s.handle_request(strat_name_request)
        self.assertEqual(msg["current_strat_name"], "opt_strat")


if __name__ == "__main__":
    unittest.main()
