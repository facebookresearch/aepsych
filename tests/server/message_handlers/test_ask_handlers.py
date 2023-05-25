#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from unittest.mock import patch

from aepsych.server.message_handlers.handle_ask import handle_ask

from ..test_server import BaseServerTestCase


class AskTestCase(BaseServerTestCase):
    @patch(
        "aepsych.server.server.handle_ask",
        return_value="handle_ask_called",
    )
    def test_unversioned_handler_types_ask(
        self, _mock_handle_ask
    ):  # TODO: edited this test
        """test_unversioned_handler_types_ask"""
        request = {"test": "test request"}
        request["type"] = "ask"
        result = self.s.unversioned_handler(request)
        self.assertEqual("handle_ask_called", result)

    @patch(
        "aepsych.server.message_handlers.handle_ask.ask",
        return_value="ask success",
    )
    def test_handle_ask(self, _mock_ask):
        """test_handle_ask - Doesn't mock the db, this will create a real db entry"""
        request = {"test": "test request"}

        self.dummy_create_setup(self.s)

        result = handle_ask(self.s, request)
        self.assertEqual("ask success", result)


if __name__ == "__main__":
    unittest.main()
