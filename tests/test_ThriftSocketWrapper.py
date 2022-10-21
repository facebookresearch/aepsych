#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import unittest
import uuid
from queue import Queue
from unittest.mock import MagicMock

import aepsych.server as server
import aepsych.utils_logging as utils_logging
from parameterized import parameterized


class ThriftSocketTestCase(unittest.TestCase):
    def setUp(self):
        # setup logger
        server.logger = utils_logging.getLogger(logging.DEBUG, "logs")
        socket = server.sockets.ThriftSocketWrapper(msg_queue=Queue())
        # random datebase path name without dashes
        database_path = "./{}.db".format(str(uuid.uuid4().hex))
        self.s = server.AEPsychServer(
            socket=socket, database_path=database_path, thrift=True
        )

    def tearDown(self):
        # cleanup the db
        if self.s.db is not None:
            self.s.db.delete_db()

    def test_serve(self):
        request = {"version": 0}
        self.s.socket.receive = MagicMock(return_value=request)
        self.s.versioned_handler = MagicMock()
        self.s.unversioned_handler = MagicMock()
        self.s.serve()
        self.s.versioned_handler.assert_called_once_with(request)

        request = {}
        self.s.socket.receive = MagicMock(return_value=request)
        self.s.versioned_handler = MagicMock()
        self.s.unversioned_handler = MagicMock()
        self.s.serve()
        self.s.unversioned_handler.assert_called_once_with(request)

    def test_receive(self):
        request = {"version": 0}
        self.s.socket.msg_queue.put(request, block=True)
        res = self.s.socket.receive()
        self.assertEqual(res, request)

    @parameterized.expand(
        [
            ({"key1": 1.0, "key2": 2.0}, json.dumps({"key1": 1.0, "key2": 2.0})),
            ("test", "test"),
            (1, "1"),
        ]
    )
    def test_send(self, test_messages, expected_results):
        self.s.socket.send(test_messages)
        server.SimplifyArrays = MagicMock(return_value=test_messages)
        self.assertEqual(self.s.socket.msg_queue.get(), expected_results)


if __name__ == "__main__":
    unittest.main()
