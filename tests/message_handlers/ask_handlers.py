#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
import uuid
from unittest.mock import MagicMock

from unittest.mock import patch
import aepsych.utils_logging as utils_logging

from aepsych.config import Config

from aepsych.server.message_handlers.handle_setup import configure

from aepsych.server.message_handlers.handle_ask import handle_ask
import aepsych.server as server

class MessageHandlerAskTests(unittest.TestCase):
    def setUp(self):
        # setup logger
        server.logger = utils_logging.getLogger(logging.DEBUG, "logs")
        # random port
        socket = server.sockets.PySocket(port=0)
        # random datebase path name without dashes
        database_path = "./{}.db".format(str(uuid.uuid4().hex))
        self.s = server.AEPsychServer(socket=socket, database_path=database_path)

    def tearDown(self):
        self.s.cleanup()

        # cleanup the db
        if self.s.db is not None:
            self.s.db.delete_db()

    def dummy_create_setup(self, server, request=None):
        request = request or {"test": "test request"}
        server._db_master_record = server.db.record_setup(
            description="default description", name="default name", request=request
        )

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
    def test_handle_ask(self, _mock_ask):  # TODO: edited this test
        """test_handle_ask - Doesn't mock the db, this will create a real db entry"""
        request = {"test": "test request"}

        self.dummy_create_setup(self.s)

        result = handle_ask(self.s, request)
        self.assertEqual("ask success", result)


if __name__ == "__main__":
    unittest.main()
