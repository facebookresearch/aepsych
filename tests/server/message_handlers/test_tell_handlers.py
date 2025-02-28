#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest
from unittest.mock import MagicMock

from ..test_server import BaseServerTestCase, dummy_config


class MessageHandlerTellTests(BaseServerTestCase):
    def test_tell(self):
        setup_request = {
            "type": "setup",
            "message": {"config_str": dummy_config},
        }

        tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5]}, "outcome": 1},
        }

        self.s.db.record_message = MagicMock()

        self.s.handle_request(setup_request)
        msg = self.s.handle_request(tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 1)
        self.assertEqual(len(self.s.strat.x), 1)
        self.assertEqual(msg["trials_recorded"], 1)
        self.assertEqual(msg["model_data_added"], 1)

        tell_request["message"]["model_data"] = False
        msg = self.s.handle_request(tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 2)
        self.assertEqual(len(self.s.strat.x), 1)
        self.assertEqual(msg["trials_recorded"], 1)
        self.assertEqual(msg["model_data_added"], 0)

    def test_batch_tell(self):
        setup_request = {
            "type": "setup",
            "message": {"config_str": dummy_config},
        }

        batch_tell_request = {
            "type": "tell",
            "message": {"config": {"x": [[0.5], [1.0], [0.0]]}, "outcome": [1, 0, 1]},
        }

        self.s.db.record_message = MagicMock()

        self.s.handle_request(setup_request)
        msg = self.s.handle_request(batch_tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 1)
        self.assertEqual(len(self.s.strat.x), 3)
        self.assertEqual(msg["trials_recorded"], 3)
        self.assertEqual(msg["model_data_added"], 3)

        self.s.handle_request(batch_tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 2)
        self.assertEqual(len(self.s.strat.x), 6)

        batch_tell_request["message"]["model_data"] = False
        msg = self.s.handle_request(batch_tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 3)
        self.assertEqual(len(self.s.strat.x), 6)
        self.assertEqual(msg["trials_recorded"], 3)
        self.assertEqual(msg["model_data_added"], 0)

    def test_tell_extra_data(self):
        setup_request = {
            "type": "setup",
            "message": {"config_str": dummy_config},
        }

        tell_request = {
            "type": "tell",
            "message": {
                "config": {"x": [0.5]},
                "outcome": 1,
                "extra": "data",
                "additional": 1,
            },
        }

        self.s.db.record_message = MagicMock()

        self.s.handle_request(setup_request)
        self.s.handle_request(tell_request)

        raw_row = self.s.db.get_raw_for(1)[0]
        extra_data = json.loads(raw_row.extra_data)
        self.assertTrue(extra_data["extra"] == "data")
        self.assertTrue(extra_data["additional"] == 1)


if __name__ == "__main__":
    unittest.main()
