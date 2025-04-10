#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

from ..test_server import BaseServerTestCase, dummy_config


class HandleExitTestCase(BaseServerTestCase):
    def test_handle_exit(self):
        request = {}
        request["type"] = "exit"
        self.s.socket.accept_client = MagicMock()
        self.s.socket.receive = MagicMock(return_value=request)
        self.s.dump = MagicMock()

        with self.assertLogs() as log:
            with self.assertRaises(SystemExit) as cm:
                self.s.serve()

        self.assertIn("No connection to send to!", " ".join(log.output))
        self.assertEqual(cm.exception.code, 0)

    def test_exit_response(self):
        setup_request = {
            "type": "setup",
            "message": {"config_str": dummy_config},
        }

        tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5]}, "outcome": 1},
        }

        exit_request = {"type": "exit"}

        self.s.db.record_message = MagicMock()

        self.s.handle_request(setup_request)
        self.s.handle_request(tell_request)
        response = self.s.handle_request(exit_request)

        self.assertEqual(response["termination_type"], "Terminate")
        self.assertTrue(response["success"])


if __name__ == "__main__":
    unittest.main()
