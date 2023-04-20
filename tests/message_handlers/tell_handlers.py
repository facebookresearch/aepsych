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


dummy_config = """
[common]
lb = [0]
ub = [1]
parnames = [x]
stimuli_per_trial = 1
outcome_types = [binary]
strategy_names = [init_strat, opt_strat]

[init_strat]
min_asks = 2
generator = SobolGenerator
min_total_outcome_occurrences = 0

[opt_strat]
min_asks = 2
generator = OptimizeAcqfGenerator
acqf = MCPosteriorVariance
model = GPClassificationModel
min_total_outcome_occurrences = 0

[GPClassificationModel]
inducing_size = 10
mean_covar_factory = default_mean_covar_factory

[SobolGenerator]
n_points = 2
"""


class MessageHandlerTellTests(unittest.TestCase):
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

    # def test_unversioned_handler_types_tell(self): #TODO: edited test
    #     """test_unversioned_handler_types_tell"""
    #     request = {"message": {"target": "test request"}}
    #     self.s.handle_setup = MagicMock(return_value=True)

    #     request["type"] = "tell"
    #     handle_tell = MagicMock(return_value=True)
    #     result = self.s.unversioned_handler(request)
    #     self.assertEqual(True, result)

    # def test_handle_tell(self): #TODO: edited test
    #     """test_handle_tell - Doesn't mock the db, this will create a real db entry"""
    #     request = {"message": {"target": "test request"}}

    #     self.s.tell = MagicMock(return_value="ask success")
    #     self.dummy_create_setup(self.s)

    #     result = handle_tell(self.s, request)
    #     self.assertEqual("acq", result)

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
