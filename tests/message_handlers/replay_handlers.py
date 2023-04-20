#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
import uuid
from unittest.mock import MagicMock, call, patch

import aepsych.server as server
import aepsych.utils_logging as utils_logging
import torch

from aepsych.server.message_handlers.handle_replay import replay

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


class MessageHandlerReplayTests(unittest.TestCase):
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

    def test_replay_order(self):
        """test_replay - verifies the replay is working, uses a test db version but does some
        amount of integration work.
        """
        request = {"message": {"target": "test request"}}
        # 1. create setup then send some messages through
        self.dummy_create_setup(self.s)
        self.s.db.record_message(
            master_table=self.s._db_master_record, type="tell", request=request
        )
        self.s.db.record_message(
            master_table=self.s._db_master_record, type="ask", request=request
        )
        self.s.db.record_message(
            master_table=self.s._db_master_record, type="update", request=request
        )
        self.s.db.record_message(
            master_table=self.s._db_master_record, type="end", request=request
        )

        replay_records = self.s.db.get_replay_for(
            self.s._db_master_record.experiment_id
        )

        self.assertEqual(len(replay_records), 5)
        self.assertEqual(replay_records[0].message_type, "setup")
        self.assertEqual(replay_records[1].message_type, "tell")
        self.assertEqual(replay_records[2].message_type, "ask")
        self.assertEqual(replay_records[3].message_type, "update")
        self.assertEqual(replay_records[4].message_type, "end")

    def test_replay_server_none_uuid(self):
        """test_replay_server_failed_uuid - check expected behavior on None UUID"""
        self.assertRaises(RuntimeError, replay, self.s, None)

    def test_replay_server_none_db(self):
        # Remove the db generated from setUp()
        self.s.db.delete_db()
        self.s.db = None
        self.assertRaises(RuntimeError, replay, self.s, "TEST")

    def test_replay_server_uuid_fails(self):
        self.assertRaises(RuntimeError, replay, self.s, "TEST")

    def test_replay_server_func(self):
        test_calls = []
        request = {"message": {"target": "setup"}, "type": "setup"}
        # 1. create setup then send some messages through
        self.dummy_create_setup(self.s, request)
        test_calls.append(call(request))

        request = {"message": {"target": "tell"}, "type": "tell"}
        self.s.db.record_message(
            master_table=self.s._db_master_record, type="tell", request=request
        )
        test_calls.append(call(request))

        request = {"message": {"target": "ask"}, "type": "ask"}
        self.s.db.record_message(
            master_table=self.s._db_master_record, type="ask", request=request
        )
        test_calls.append(call(request))

        request = {"message": {"target": "update"}, "type": "update"}
        self.s.db.record_message(
            master_table=self.s._db_master_record, type="update", request=request
        )
        test_calls.append(call(request))

        self.s.unversioned_handler = MagicMock()
        replay(self.s, self.s._db_master_record.experiment_id)
        print(f"replay called with check = {test_calls}")
        self.s.unversioned_handler.assert_has_calls(test_calls, any_order=False)

    @patch(
        "aepsych.strategy.Strategy.gen",
        side_effect=lambda num_points: torch.tensor([[0.0]]),
    )
    @patch(
        "aepsych.strategy.Strategy.get_max",
        side_effect=lambda query_type: (torch.tensor([[0.0]]), torch.tensor([[0.0]])),
    )
    def test_replay_server_skip_computations(self, mock_query, mock_gen):
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": dummy_config},
        }
        self.s.versioned_handler(setup_request)

        ask_request = {"type": "ask", "message": ""}

        tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5]}, "outcome": 1},
        }
        self.s.unversioned_handler(tell_request)

        n_asks = 3  # number of asks needed to move to opt_strat
        for _ in range(n_asks):
            self.s.unversioned_handler(ask_request)

        query_request = {
            "type": "query",
            "message": {
                "query_type": "max",
            },
        }
        self.s.unversioned_handler(query_request)

        replay(self.s, self.s._db_master_record.experiment_id, skip_computations=True)

        # gen and query are called once each from the messages above
        self.assertEqual(mock_gen.call_count, n_asks)
        self.assertEqual(mock_query.call_count, 1)
        self.assertEqual(self.s.strat._strat_idx, 1)  # make sure init_strat finished

        replay(self.s, self.s._db_master_record.experiment_id, skip_computations=False)
        self.assertEqual(mock_gen.call_count, n_asks * 2)
        self.assertEqual(mock_query.call_count, 2)


if __name__ == "__main__":
    unittest.main()
