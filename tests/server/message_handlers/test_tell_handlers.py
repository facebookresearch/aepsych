#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

from aepsych.server.message_handlers.handle_tell import handle_tell

from ..test_server import BaseServerTestCase, dummy_config


class MessageHandlerTellTests(BaseServerTestCase):
    @patch(
        "aepsych.server.server.handle_tell",
        return_value="handle_tell_called",
    )
    def test_unversioned_handler_types_tell(
        self, _mock_handle_tell
    ):  # TODO: edited this test
        """test_unversioned_handler_types_tell"""
        request = {"message": {"target": "test request"}}
        self.s.handle_setup = MagicMock(return_value=True)

        request["type"] = "tell"
        result = self.s.unversioned_handler(request)
        self.assertEqual("handle_tell_called", result)

    @patch(
        "aepsych.server.message_handlers.handle_tell.tell",
        return_value="tell_called",
    )
    def test_handle_tell(self, _mock_tell):  # TODO: edited this test
        """test_handle_tell - Doesn't mock the db, this will create a real db entry"""
        request = {"message": {"target": "test request"}}

        self.s.tell = MagicMock(return_value="ask success")
        self.dummy_create_setup(self.s)

        result = handle_tell(self.s, request)
        self.assertEqual("acq", result)

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

        self.s.unversioned_handler(setup_request)
        self.s.unversioned_handler(tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 2)
        self.assertEqual(len(self.s.strat.x), 1)

        tell_request["message"]["model_data"] = False
        self.s.unversioned_handler(tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 3)
        self.assertEqual(len(self.s.strat.x), 1)


if __name__ == "__main__":
    unittest.main()
