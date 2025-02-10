#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import unittest

from ..test_server import AsyncServerTestBase, dummy_config


class HandleExitTestCase(AsyncServerTestBase):
    async def test_handle_exit(self):
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": dummy_config},
        }

        await self.mock_client(setup_request)

        request = {}
        request["type"] = "exit"
        await self.mock_client(request)

        with self.assertRaises(ConnectionRefusedError):
            await asyncio.open_connection(self.s.host, self.s.port)

        self.assertTrue(self.s.exit_server_loop)


if __name__ == "__main__":
    unittest.main()
