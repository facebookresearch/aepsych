#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import select
import time
import unittest
import uuid
from unittest.mock import MagicMock

import aepsych.server as server
import aepsych.utils_logging as utils_logging

from aepsych.server.sockets import BAD_REQUEST

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


class BaseServerTestCase(unittest.TestCase):
    def setUp(self):
        # setup logger
        server.logger = utils_logging.getLogger(logging.DEBUG, "logs")
        # random port
        socket = server.sockets.PySocket(port=0)
        # random datebase path name without dashes
        database_path = "./{}_test_server.db".format(str(uuid.uuid4().hex))
        self.s = server.AEPsychServer(socket=socket, database_path=database_path)
        self.db_name = database_path.split("/")[1]
        self.db_path = database_path

    def tearDown(self):
        self.s.cleanup()

        # sleep to ensure db is closed
        time.sleep(0.2)

        # cleanup the db
        if self.s.db is not None:
            try:
                self.s.db.delete_db()
            except PermissionError as e:
                print("Failed to deleted database: ", e)

    def dummy_create_setup(self, server, request=None):
        request = request or {"test": "test request"}
        server._db_master_record = server.db.record_setup(
            description="default description", name="default name", request=request
        )


class ServerTestCase(BaseServerTestCase):
    def test_final_strat_serialization(self):
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": dummy_config},
        }
        ask_request = {"type": "ask", "message": ""}
        tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5]}, "outcome": 1},
        }
        self.s.handle_request(setup_request)
        while not self.s.strat.finished:
            self.s.handle_request(ask_request)
            self.s.handle_request(tell_request)

        exp_id = self.s.db.get_master_records()[-1].experiment_id
        stored_strat = self.s.get_strat_from_replay(exp_id)
        # just some spot checks that the strat's the same
        # same data. We do this twice to make sure buffers are
        # in a good state and we can load twice without crashing
        for _ in range(2):
            stored_strat = self.s.get_strat_from_replay(exp_id)
            self.assertTrue((stored_strat.x == self.s.strat.x).all())
            self.assertTrue((stored_strat.y == self.s.strat.y).all())
            # same lengthscale and outputscale
            self.assertEqual(
                stored_strat.model.covar_module.lengthscale,
                self.s.strat.model.covar_module.lengthscale,
            )

    def test_pandadf_dump_single(self):
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": dummy_config},
        }
        ask_request = {"type": "ask", "message": ""}
        tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5]}, "outcome": 1},
            "extra_info": {},
        }
        self.s.handle_request(setup_request)
        expected_x = [0, 1, 2, 3]
        expected_z = list(reversed(expected_x))
        expected_y = [x % 2 for x in expected_x]
        i = 0
        while not self.s.strat.finished:
            self.s.handle_request(ask_request)
            tell_request["message"]["config"]["x"] = [expected_x[i]]
            tell_request["message"]["config"]["z"] = [expected_z[i]]
            tell_request["message"]["outcome"] = expected_y[i]
            tell_request["extra_info"]["e1"] = 1
            tell_request["extra_info"]["e2"] = 2
            i = i + 1
            self.s.handle_request(tell_request)

        exp_id = self.s.db.get_master_records()[-1].experiment_id
        out_df = self.s.get_dataframe_from_replay(exp_id)
        self.assertTrue((out_df.x == expected_x).all())
        self.assertTrue((out_df.z == expected_z).all())
        self.assertTrue((out_df.response == expected_y).all())
        self.assertTrue((out_df.e1 == [1] * 4).all())
        self.assertTrue((out_df.e2 == [2] * 4).all())
        self.assertTrue("post_mean" in out_df.columns)
        self.assertTrue("post_var" in out_df.columns)

    def test_pandadf_dump_multistrat(self):
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": dummy_config},
        }
        ask_request = {"type": "ask", "message": ""}
        tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5]}, "outcome": 1},
            "extra_info": {},
        }
        expected_x = [0, 1, 2, 3]
        expected_z = list(reversed(expected_x))
        expected_y = [x % 2 for x in expected_x]
        i = 0
        self.s.handle_request(setup_request)
        while not self.s.strat.finished:
            self.s.handle_request(ask_request)
            tell_request["message"]["config"]["x"] = [expected_x[i]]
            tell_request["message"]["config"]["z"] = [expected_z[i]]
            tell_request["message"]["outcome"] = expected_y[i]
            tell_request["extra_info"]["e1"] = 1
            tell_request["extra_info"]["e2"] = 2
            i = i + 1
            self.s.handle_request(tell_request)

        exp_id = self.s.db.get_master_records()[-1].experiment_id
        out_df = self.s.get_dataframe_from_replay(exp_id)

        self.assertTrue((out_df.x == expected_x).all())
        self.assertTrue((out_df.z == expected_z).all())
        self.assertTrue((out_df.response == expected_y).all())
        self.assertTrue((out_df.e1 == [1] * len(expected_x)).all())
        self.assertTrue((out_df.e2 == [2] * len(expected_x)).all())
        self.assertTrue("post_mean" in out_df.columns)
        self.assertTrue("post_var" in out_df.columns)

    def test_pandadf_dump_flat(self):
        """
        This test handles the case where the config values are flat
        scalars and not lists
        """
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": dummy_config},
        }
        ask_request = {"type": "ask", "message": ""}
        tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5]}, "outcome": 1},
            "extra_info": {},
        }
        self.s.handle_request(setup_request)
        expected_x = [0, 1, 2, 3]
        expected_z = list(reversed(expected_x))
        expected_y = [x % 2 for x in expected_x]
        i = 0
        while not self.s.strat.finished:
            self.s.handle_request(ask_request)
            tell_request["message"]["config"]["x"] = expected_x[i]
            tell_request["message"]["config"]["z"] = expected_z[i]
            tell_request["message"]["outcome"] = expected_y[i]
            tell_request["extra_info"]["e1"] = 1
            tell_request["extra_info"]["e2"] = 2
            i = i + 1
            self.s.handle_request(tell_request)

        exp_id = self.s.db.get_master_records()[-1].experiment_id
        out_df = self.s.get_dataframe_from_replay(exp_id)
        self.assertTrue((out_df.x == expected_x).all())
        self.assertTrue((out_df.z == expected_z).all())
        self.assertTrue((out_df.response == expected_y).all())
        self.assertTrue((out_df.e1 == [1] * 4).all())
        self.assertTrue((out_df.e2 == [2] * 4).all())
        self.assertTrue("post_mean" in out_df.columns)
        self.assertTrue("post_var" in out_df.columns)

    def test_receive(self):
        """test_receive - verifies the receive is working when server receives unexpected messages"""

        message1 = b"\x16\x03\x01\x00\xaf\x01\x00\x00\xab\x03\x03\xa9\x80\xcc"  # invalid message
        message2 = b"\xec\xec\x14M\xfb\xbd\xac\xe7jF\xbe\xf9\x9bM\x92\x15b\xb5"  # invalid message
        message3 = {"message": {"target": "test request"}}  # valid message
        message_list = [message1, message2, json.dumps(message3)]

        self.s.socket.conn = MagicMock()

        for i, message in enumerate(message_list):
            select.select = MagicMock(return_value=[[self.s.socket.conn], [], []])
            self.s.socket.conn.recv = MagicMock(return_value=message)
            if i != 2:
                self.assertEqual(self.s.socket.receive(False), BAD_REQUEST)
            else:
                self.assertEqual(self.s.socket.receive(False), message3)

    def test_error_handling(self):
        # double brace escapes, single brace to substitute, so we end up with 3 braces
        request = f"{{{BAD_REQUEST}}}"

        expected_error = f"server_error, Request '{request}' raised error ''str' object has no attribute 'keys''!"

        self.s.socket.accept_client = MagicMock()

        self.s.socket.receive = MagicMock(return_value=request)
        self.s.socket.send = MagicMock()
        self.s.exit_server_loop = True
        with self.assertRaises(SystemExit):
            self.s.serve()
        self.s.socket.send.assert_called_once_with(expected_error)

    def test_queue(self):
        """Test to see that the queue is being handled correctly"""

        self.s.socket.accept_client = MagicMock()
        ask_request = {"type": "ask", "message": ""}
        self.s.socket.receive = MagicMock(return_value=ask_request)
        self.s.socket.send = MagicMock()
        self.s.exit_server_loop = True
        with self.assertRaises(SystemExit):
            self.s.serve()
        assert len(self.s.queue) == 0


if __name__ == "__main__":
    unittest.main()
