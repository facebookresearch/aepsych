#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from aepsych.config import Config

from ..test_server import BaseServerTestCase, dummy_config


class HandleExitTestCase(BaseServerTestCase):
    def test_get_config(self):
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": dummy_config},
        }
        get_config_request = {"type": "get_config", "message": {}}

        self.s.handle_request(setup_request)
        config_dict = self.s.handle_request(get_config_request)
        true_config_dict = Config(config_str=dummy_config).to_dict(deduplicate=False)
        self.assertEqual(config_dict, true_config_dict)

        get_config_request["message"] = {
            "section": "init_strat",
            "property": "min_asks",
        }
        response = self.s.handle_request(get_config_request)
        self.assertEqual(response, true_config_dict["init_strat"]["min_asks"])

        get_config_request["message"] = {"section": "init_strat", "property": "lb"}
        response = self.s.handle_request(get_config_request)
        self.assertEqual(response, true_config_dict["init_strat"]["lb"])

        get_config_request["message"] = {"property": "min_asks"}
        with self.assertRaises(RuntimeError):
            response = self.s.handle_request(get_config_request)

        get_config_request["message"] = {"section": "init_strat"}
        with self.assertRaises(RuntimeError):
            response = self.s.handle_request(get_config_request)


if __name__ == "__main__":
    unittest.main()
