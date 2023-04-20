#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
import uuid
from unittest.mock import MagicMock

import aepsych.server as server
import aepsych.utils_logging as utils_logging
class MessageHandlerTests(unittest.TestCase):
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
  
    def test_unversioned_handler_untyped(self):
        """test_unversioned_handler_untyped"""
        request = {}
        # check untyped request
        with self.assertRaises(RuntimeError):
            self.s.unversioned_handler(request)

    def test_unversioned_handler_type_invalid(self):
        """test_unversioned_handler_type_invalid"""
        request = {"type": "invalid"}
        # make sure invalid types handle properly
        with self.assertRaises(RuntimeError):
            self.s.unversioned_handler(request)

    def test_unversioned_handler_types_update(self):
        """test_unversioned_handler_types_update"""
        request = {}
        # self.s.handle_setup = MagicMock(return_value=True)

        request["type"] = "update"
        self.s.handle_update = MagicMock(return_value=True)
        result = self.s.unversioned_handler(request)
        self.assertEqual(True, result)

    def test_v01_handler_types_resume(self):
        """test setup v01"""
        request = {}
        self.s.handle_resume_v01 = MagicMock(return_value=True)
        self.s.socket.send = MagicMock()
        request["type"] = "resume"
        request["version"] = "0.01"
        result = self.s.versioned_handler(request)
        self.assertEqual(True, result)

    def test_serve_versioned_handler(self):
        """Tests that the full pipeline is working. Message should go from _receive_send to _handle_queue
        to the version handler"""
        request = {"version": 0}
        self.s.socket.receive = MagicMock(return_value=request)
        self.s.socket.accept_client = MagicMock()

        self.s.versioned_handler = MagicMock()
        self.s.unversioned_handler = MagicMock()
        self.s.exit_server_loop = True
        with self.assertRaises(SystemExit):
            self.s.serve()

    def test_serve_unversioned_handler(self):
        request = {}
        self.s.socket.receive = MagicMock(return_value=request)
        self.s.socket.accept_client = MagicMock()

        self.s.versioned_handler = MagicMock()
        self.s.unversioned_handler = MagicMock()
        self.s.exit_server_loop = True
        with self.assertRaises(SystemExit):
            self.s.serve()


if __name__ == "__main__":
    unittest.main()
