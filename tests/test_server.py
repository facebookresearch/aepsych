#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import unittest
import uuid
from unittest.mock import MagicMock, call, patch

import aepsych.server as server
import aepsych.utils_logging as utils_logging


dummy_config = """
[common]
lb = [0]
ub = [1]
parnames = [x]
outcome_type = single_probit
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


class ServerTestCase(unittest.TestCase):
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

    def test_unversioned_handler_types_ask(self):
        """test_unversioned_handler_types_ask"""
        request = {}

        request["type"] = "ask"
        self.s.handle_ask = MagicMock(return_value=True)
        result = self.s.unversioned_handler(request)
        self.assertEqual(True, result)

    def test_unversioned_handler_types_tell(self):
        """test_unversioned_handler_types_tell"""
        request = {}
        # self.s.handle_setup = MagicMock(return_value=True)

        request["type"] = "tell"
        self.s.handle_tell = MagicMock(return_value=True)
        result = self.s.unversioned_handler(request)
        self.assertEqual(True, result)

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

    def dummy_create_setup(self, server, request=None):
        request = request or {"test": "test request"}
        server._db_master_record = server.db.record_setup(
            description="default description", name="default name", request=request
        )

    def test_handle_ask(self):
        """test_handle_ask - Doesn't mock the db, this will create a real db entry"""
        request = {"test": "test request"}

        self.s.ask = MagicMock(return_value="ask success")

        self.dummy_create_setup(self.s)

        result = self.s.handle_ask(request)
        self.assertEqual("ask success", result)

    def test_handle_tell(self):
        """test_handle_tell - Doesn't mock the db, this will create a real db entry"""
        request = {"message": {"target": "test request"}}

        self.s.tell = MagicMock(return_value="ask success")
        self.dummy_create_setup(self.s)

        result = self.s.handle_tell(request)
        self.assertEqual("acq", result)

    def test_handle_update(self):
        """test_handle_update - Doesn't mock the db, this will create a real db entry"""
        request = {"message": {"target": "test request"}}

        self.s.tell = MagicMock(return_value="update success")
        self.s.ask = MagicMock(return_value="update success")
        self.s.strat = MagicMock()

        self.dummy_create_setup(self.s)

        result = self.s.handle_update(request)
        self.assertEqual("update success", result)

    def test_handle_exit(self):
        request = {}
        request["type"] = "exit"
        self.s.socket.receive = MagicMock(return_value=request)
        self.s.dump = MagicMock()

        with self.assertRaises(SystemExit) as cm:
            self.s.serve()

        self.assertEqual(cm.exception.code, 0)

    @patch("socket.socket.accept")
    def test_receive(self, mock_accept):
        """test_receive - verifies the receive is working when server receives unexpected messages"""
        conn = MagicMock()
        mock_accept.return_value = (conn, MagicMock())

        message1 = b"\x16\x03\x01\x00\xaf\x01\x00\x00\xab\x03\x03\xa9\x80\xcc"  # invalid message
        message2 = b"\xec\xec\x14M\xfb\xbd\xac\xe7jF\xbe\xf9\x9bM\x92\x15b\xb5"  # invalid message
        message3 = {"message": {"target": "test request"}}  # valid message

        conn.recv = MagicMock()
        conn.recv.side_effect = iter([message1, message2, json.dumps(message3)])
        self.assertEqual(self.s.socket.receive(), message3)

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
        self.assertRaises(RuntimeError, self.s.replay, None)

    def test_replay_server_none_db(self):
        # Remove the db generated from setUp()
        self.s.db.delete_db()
        self.s.db = None
        self.assertRaises(RuntimeError, self.s.replay, "TEST")

    def test_replay_server_uuid_fails(self):
        self.assertRaises(RuntimeError, self.s.replay, "TEST")

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
        self.s.replay(self.s._db_master_record.experiment_id)
        print(f"replay called with check = {test_calls}")
        self.s.unversioned_handler.assert_has_calls(test_calls, any_order=False)

    def test_replay_server_skip_computations(self):
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

        request = {"message": {"target": "update"}, "type": "update"}
        self.s.db.record_message(
            master_table=self.s._db_master_record, type="update", request=request
        )
        test_calls.append(call(request))

        request = {"message": {"target": "plot"}, "type": "plot"}
        self.s.db.record_message(
            master_table=self.s._db_master_record, type="plot", request=request
        )
        test_calls.append(call(request))

        self.s.unversioned_handler = MagicMock()

        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": dummy_config},
        }
        self.s.versioned_handler(setup_request)
        self.s.strat.has_model = False

        self.s.replay(self.s._db_master_record.experiment_id, skip_computations=True)
        print(f"replay called with check = {test_calls}")
        self.s.unversioned_handler.assert_has_calls(test_calls, any_order=False)
        self.assertEqual(self.s.unversioned_handler.call_count, len(test_calls))

    def test_serve_versioned_handler(self):
        request = {"version": 0}
        self.s.socket.receive = MagicMock(return_value=request)
        self.s.versioned_handler = MagicMock()
        self.s.unversioned_handler = MagicMock()
        self.s.exit_server_loop = True
        with self.assertRaises(SystemExit):
            self.s.serve()
        self.s.versioned_handler.assert_called_once_with(request)

    def test_serve_unversioned_handler(self):
        request = {}
        self.s.socket.receive = MagicMock(return_value=request)
        self.s.versioned_handler = MagicMock()
        self.s.unversioned_handler = MagicMock()
        self.s.exit_server_loop = True
        with self.assertRaises(SystemExit):
            self.s.serve()
        self.s.unversioned_handler.assert_called_once_with(request)

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
        self.s.versioned_handler(setup_request)
        while not self.s.strat.finished:
            self.s.unversioned_handler(ask_request)
            self.s.unversioned_handler(tell_request)

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
            self.assertEqual(
                stored_strat.model.covar_module.outputscale,
                self.s.strat.model.covar_module.outputscale,
            )

    def test_strat_can_model(self):
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
            ],
        }
        can_model_request = {
            "type": "can_model",
            "message": {},
        }

        self.s.versioned_handler(setup_request)
        # At the start there is no model, so can_model returns false
        response = self.s.unversioned_handler(can_model_request)
        self.assertTrue(response["can_model"] == 0)

        self.s.unversioned_handler(ask_request)
        self.s.unversioned_handler(tell_request)
        self.s.unversioned_handler(ask_request)
        self.s.unversioned_handler(tell_request)
        self.s.unversioned_handler(ask_request)

        # Dummy config has 2 init trials; so after third ask, can_model returns true
        response = self.s.unversioned_handler(can_model_request)
        self.assertTrue(response["can_model"] == 1)

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
        self.s.versioned_handler(setup_request)
        expected_x = [0, 1, 2, 3]
        expected_z = list(reversed(expected_x))
        expected_y = [x % 2 for x in expected_x]
        i = 0
        while not self.s.strat.finished:
            self.s.unversioned_handler(ask_request)
            tell_request["message"]["config"]["x"] = [expected_x[i]]
            tell_request["message"]["config"]["z"] = [expected_z[i]]
            tell_request["message"]["outcome"] = expected_y[i]
            tell_request["extra_info"]["e1"] = 1
            tell_request["extra_info"]["e2"] = 2
            i = i + 1
            self.s.unversioned_handler(tell_request)

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
        expected_x = [0, 1, 2, 3] * 2
        expected_z = list(reversed(expected_x))
        expected_y = [x % 2 for x in expected_x]
        for _ in range(2):
            i = 0
            self.s.versioned_handler(setup_request)
            while not self.s.strat.finished:
                self.s.unversioned_handler(ask_request)
                tell_request["message"]["config"]["x"] = [expected_x[i]]
                tell_request["message"]["config"]["z"] = [expected_z[i]]
                tell_request["message"]["outcome"] = expected_y[i]
                tell_request["extra_info"]["e1"] = 1
                tell_request["extra_info"]["e2"] = 2
                i = i + 1
                self.s.unversioned_handler(tell_request)

        exp_id = self.s.db.get_master_records()[-1].experiment_id
        out_df = self.s.get_dataframe_from_replay(exp_id)

        self.assertTrue((out_df.x == expected_x).all())
        self.assertTrue((out_df.z == expected_z).all())
        self.assertTrue((out_df.response == expected_y).all())
        self.assertTrue((out_df.e1 == [1] * 8).all())
        self.assertTrue((out_df.e2 == [2] * 8).all())
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
        self.s.versioned_handler(setup_request)
        expected_x = [0, 1, 2, 3]
        expected_z = list(reversed(expected_x))
        expected_y = [x % 2 for x in expected_x]
        i = 0
        while not self.s.strat.finished:
            self.s.unversioned_handler(ask_request)
            tell_request["message"]["config"]["x"] = expected_x[i]
            tell_request["message"]["config"]["z"] = expected_z[i]
            tell_request["message"]["outcome"] = expected_y[i]
            tell_request["extra_info"]["e1"] = 1
            tell_request["extra_info"]["e2"] = 2
            i = i + 1
            self.s.unversioned_handler(tell_request)

        exp_id = self.s.db.get_master_records()[-1].experiment_id
        out_df = self.s.get_dataframe_from_replay(exp_id)
        self.assertTrue((out_df.x == expected_x).all())
        self.assertTrue((out_df.z == expected_z).all())
        self.assertTrue((out_df.response == expected_y).all())
        self.assertTrue((out_df.e1 == [1] * 4).all())
        self.assertTrue((out_df.e2 == [2] * 4).all())
        self.assertTrue("post_mean" in out_df.columns)
        self.assertTrue("post_var" in out_df.columns)

    def test_error_handling(self):
        request = {"bad request"}
        self.s.socket.receive = MagicMock(return_value=request)
        self.s.socket.send = MagicMock()
        self.s.exit_server_loop = True
        with self.assertRaises(SystemExit):
            self.s.serve()
        self.s.socket.send.assert_called_once_with("bad request")


if __name__ == "__main__":
    unittest.main()
