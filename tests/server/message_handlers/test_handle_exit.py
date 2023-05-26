#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

from ..test_server import BaseServerTestCase


class HandleExitTestCase(BaseServerTestCase):
    def test_handle_exit(self):
        request = {}
        request["type"] = "exit"
        self.s.socket.accept_client = MagicMock()
        self.s.socket.receive = MagicMock(return_value=request)
        self.s.dump = MagicMock()

        with self.assertRaises(SystemExit) as cm:
            self.s.serve()

        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
