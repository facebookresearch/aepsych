#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import select
import unittest
import uuid
import torch
from unittest.mock import MagicMock, patch, PropertyMock

import aepsych.server as server
import aepsych.utils_logging as utils_logging
from aepsych.config import Config
from aepsych.server.sockets import BAD_REQUEST
from aepsych.strategy import AEPsychStrategy
from aepsych.server.message_handlers.handle_replay import (
    get_strat_from_replay,
    get_dataframe_from_replay,
)
from aepsych.server.message_handlers.handle_setup import configure

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


class ServerTestCase(unittest.TestCase):
    def setUp(self):
        # setup logger
        server.logger = utils_logging.getLogger(logging.DEBUG, "logs")
        # random port
        socket = server.sockets.PySocket(port=0)
        # random datebase path name without dashes
        database_path = "./{}_test_server.db".format(str(uuid.uuid4().hex))
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
        "aepsych.server.message_handlers.handle_ask.ask", return_value="update success"
    )
    @patch(
        "aepsych.server.message_handlers.handle_tell.tell",
        return_value="update success",
    )
    def test_handle_update(
        self, _mock_handle_ask, _mock_handle_tell
    ):  # TODO: edited test
        """test_handle_update - Doesn't mock the db, this will create a real db entry"""
        request = {"message": {"target": "test request"}}
        self.s.strat = MagicMock()

        self.dummy_create_setup(self.s)

        result = self.s.handle_update(request)
        self.assertEqual("update success", result)

    def test_handle_exit(self):
        request = {}
        request["type"] = "exit"
        self.s.socket.accept_client = MagicMock()
        self.s.socket.receive = MagicMock(return_value=request)
        self.s.dump = MagicMock()

        with self.assertRaises(SystemExit) as cm:
            self.s.serve()

        self.assertEqual(cm.exception.code, 0)

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
        stored_strat = get_strat_from_replay(self.s, exp_id)
        # just some spot checks that the strat's the same
        # same data. We do this twice to make sure buffers are
        # in a good state and we can load twice without crashing
        for _ in range(2):
            stored_strat = get_strat_from_replay(self.s, exp_id)
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
        out_df = get_dataframe_from_replay(self.s, exp_id)
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
        out_df = get_dataframe_from_replay(self.s, exp_id)

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
        out_df = get_dataframe_from_replay(self.s, exp_id)
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

    def test_handle_finish_strategy(self):
        setup_request = {
            "type": "setup",
            "message": {"config_str": dummy_config},
        }

        tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5]}, "outcome": 1},
        }

        ask_request = {"type": "ask", "message": ""}

        strat_name_request = {"type": "strategy_name"}
        finish_strat_request = {"type": "finish_strategy"}

        self.s.unversioned_handler(setup_request)
        strat_name = self.s.unversioned_handler(strat_name_request)
        self.assertEqual(strat_name, "init_strat")

        # model-based strategies require data
        self.s.unversioned_handler(tell_request)

        msg = self.s.unversioned_handler(finish_strat_request)
        self.assertEqual(msg, "finished strategy init_strat")

        # need to gen another trial to move to next strategy
        self.s.unversioned_handler(ask_request)

        strat_name = self.s.unversioned_handler(strat_name_request)
        self.assertEqual(strat_name, "opt_strat")

    def test_ax_functionality(self):
        config_str = """
        [common]
        use_ax = True
        lb = [0]
        ub = [1]
        parnames = [x]
        stimuli_per_trial = 1
        outcome_types = [binary]
        strategy_names = [init_strat, opt_strat]

        [init_strat]
        generator = SobolGenerator

        [opt_strat]
        generator = OptimizeAcqfGenerator
        model = ContinuousRegressionGP
        acqf = qNoisyExpectedImprovement
        """
        config = Config(config_str=config_str)
        configure(self.s, config=config)
        self.assertTrue(self.s.use_ax)
        self.assertIsInstance(self.s.strat, AEPsychStrategy)

    def test_config_to_tensor(self):
        with patch(
            "aepsych.server.AEPsychServer.parnames", new_callable=PropertyMock
        ) as mock_parnames:
            mock_parnames.return_value = ["par1", "par2", "par3"]

            # test single
            config = {"par1": 0.0, "par2": 1.0, "par3": 2.0}
            tensor = self.s._config_to_tensor(config)
            self.assertTrue(torch.equal(tensor, torch.tensor([0.0, 1.0, 2.0])))

            config = {"par1": [0.0], "par2": [1.0], "par3": [2.0]}
            tensor = self.s._config_to_tensor(config)
            self.assertTrue(torch.equal(tensor, torch.tensor([0.0, 1.0, 2.0])))

            # test pairwise
            config = {"par1": [0.0, 2.0], "par2": [1.0, 1.0], "par3": [2.0, 0.0]}
            tensor = self.s._config_to_tensor(config)
            self.assertTrue(
                torch.equal(tensor, torch.tensor([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]))
            )

    def test_get_config(self):
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": dummy_config},
        }
        get_config_request = {"type": "get_config", "message": {}}

        self.s.versioned_handler(setup_request)
        config_dict = self.s.versioned_handler(get_config_request)
        true_config_dict = Config(config_str=dummy_config).to_dict(deduplicate=False)
        self.assertEqual(config_dict, true_config_dict)

        get_config_request["message"] = {
            "section": "init_strat",
            "property": "min_asks",
        }
        response = self.s.versioned_handler(get_config_request)
        self.assertEqual(response, true_config_dict["init_strat"]["min_asks"])

        get_config_request["message"] = {"section": "init_strat", "property": "lb"}
        response = self.s.versioned_handler(get_config_request)
        self.assertEqual(response, true_config_dict["init_strat"]["lb"])

        get_config_request["message"] = {"property": "min_asks"}
        with self.assertRaises(RuntimeError):
            response = self.s.versioned_handler(get_config_request)

        get_config_request["message"] = {"section": "init_strat"}
        with self.assertRaises(RuntimeError):
            response = self.s.versioned_handler(get_config_request)

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
