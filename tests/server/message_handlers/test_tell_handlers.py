#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
        self.s.handle_request(tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 1)
        self.assertEqual(len(self.s.strat.x), 1)

        tell_request["message"]["model_data"] = False
        self.s.handle_request(tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 2)
        self.assertEqual(len(self.s.strat.x), 1)


if __name__ == "__main__":
    unittest.main()
