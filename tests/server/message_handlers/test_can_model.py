#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from ..test_server import BaseServerTestCase, dummy_config


class StratCanModelTestCase(BaseServerTestCase):
    def test_strat_can_model(self):
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
            ],
        }
        can_model_request = {
            "type": "can_model",
            "message": {},
        }

        self.s.handle_request(setup_request)
        # At the start there is no model, so can_model returns false
        response = self.s.handle_request(can_model_request)
        self.assertTrue(response["can_model"] == 0)

        self.s.handle_request(ask_request)
        self.s.handle_request(tell_request)
        self.s.handle_request(ask_request)
        self.s.handle_request(tell_request)
        self.s.handle_request(ask_request)

        # Dummy config has 2 init trials; so after third ask, can_model returns true
        response = self.s.handle_request(can_model_request)
        self.assertTrue(response["can_model"] == 1)


if __name__ == "__main__":
    unittest.main()
