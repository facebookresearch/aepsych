#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
import uuid

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


class MessageHandlerQueryTests(unittest.TestCase):
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

    def test_strat_query(self):
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": dummy_config},
        }
        ask_request = {"type": "ask", "message": ""}
        tell_request = {
            "type": "tell",
            "message": [
                {"config": {"x": [0.5]}, "outcome": 1},
                {"config": {"x": [0.0]}, "outcome": 0},
                {"config": {"x": [1]}, "outcome": 0},
            ],
        }

        self.s.versioned_handler(setup_request)
        while not self.s.strat.finished:
            self.s.unversioned_handler(ask_request)
            self.s.unversioned_handler(tell_request)

        query_max_req = {
            "type": "query",
            "message": {
                "query_type": "max",
            },
        }
        query_min_req = {
            "type": "query",
            "message": {
                "query_type": "min",
            },
        }
        query_pred_req = {
            "type": "query",
            "message": {
                "query_type": "prediction",
                "x": {"x": [0.0]},
            },
        }
        query_inv_req = {
            "type": "query",
            "message": {
                "query_type": "inverse",
                "y": 5.0,
            },
        }
        response_max = self.s.unversioned_handler(query_max_req)
        response_min = self.s.unversioned_handler(query_min_req)
        response_pred = self.s.unversioned_handler(query_pred_req)
        response_inv = self.s.unversioned_handler(query_inv_req)

        for response in [response_max, response_min, response_pred, response_inv]:
            self.assertTrue(type(response["x"]) is dict)
            self.assertTrue(len(response["x"]["x"]) == 1)
            self.assertTrue(type(response["y"]) is float)


if __name__ == "__main__":
    unittest.main()
