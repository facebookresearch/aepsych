#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import logging
import os
import select
import threading
import time
import unittest
import uuid
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import aepsych.server as server
import aepsych.utils_logging as utils_logging
import torch
from aepsych.server.sockets import BAD_REQUEST

dummy_config = """
[common]
lb = [0]
ub = [1]
parnames = [x]
stimuli_per_trial = 1
outcome_types = [binary]
strategy_names = [init_strat, opt_strat]

[metadata]
experiment_name = test experiment
experiment_description = dummy experiment to test the server
experiment_id = e1
participant_id = 101
extra = data that is arbitrary
array = [100, 1000]
date = Nov 26, 2024

[init_strat]
min_asks = 2
generator = SobolGenerator
min_total_outcome_occurrences = 0

[opt_strat]
min_asks = 2
generator = OptimizeAcqfGenerator
model = GPClassificationModel
min_total_outcome_occurrences = 0

[OptimizeAcqfGenerator]
acqf = MCPosteriorVariance

[GPClassificationModel]
inducing_size = 10
mean_covar_factory = default_mean_covar_factory

[SobolGenerator]
n_points = 2
"""

points = [[10, 10], [10, 11], [11, 10], [11, 11]]
manual_dummy_config = f"""
[common]
lb = [10, 10]
ub = [11, 11]
parnames = [par1, par2]
outcome_types = [binary]
stimuli_per_trial = 1
strategy_names = [init_strat]

[init_strat]
generator = ManualGenerator

[ManualGenerator]
points = {points}
seed = 123
"""


def handle_wait(server, request: dict[str, Any]):
    time.sleep(0.25)

    if request["message"]["terminate"]:
        server.exit_server_loop = True
    return {"queue_size": len(server.queue)}


class BaseServerTestCase(unittest.TestCase):
    # so that this can be overridden for tests that require specific databases.
    @property
    def database_path(self):
        return "./{}_test_server.db".format(str(uuid.uuid4().hex))

    def setUp(self):
        # setup logger
        server.logger = utils_logging.getLogger(logging.DEBUG, "logs")
        # random port
        socket = server.sockets.PySocket(port=0)
        # random datebase path name without dashes
        database_path = self.database_path
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

        unique_id = self.s.db.get_master_records()[-1].unique_id
        stored_strat = self.s.get_strat_from_replay(unique_id)
        # just some spot checks that the strat's the same
        # same data. We do this twice to make sure buffers are
        # in a good state and we can load twice without crashing
        for _ in range(2):
            stored_strat = self.s.get_strat_from_replay(unique_id)
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
            tell_request["message"]["e1"] = 1
            tell_request["message"]["e2"] = 2
            i = i + 1
            self.s.handle_request(tell_request)

        unique_id = self.s.db.get_master_records()[-1].unique_id
        out_df = self.s.get_dataframe_from_replay(unique_id)
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
            tell_request["message"]["e1"] = 1
            tell_request["message"]["e2"] = 2
            i = i + 1
            self.s.handle_request(tell_request)

        unique_id = self.s.db.get_master_records()[-1].unique_id
        out_df = self.s.get_dataframe_from_replay(unique_id)

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
            tell_request["message"]["e1"] = 1
            tell_request["message"]["e2"] = 2
            i = i + 1
            self.s.handle_request(tell_request)

        unique_id = self.s.db.get_master_records()[-1].unique_id
        out_df = self.s.get_dataframe_from_replay(unique_id)
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

    def test_replay(self):
        exp_config = """
            [common]
            lb = [0]
            ub = [1]
            parnames = [x]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [manual_strat, sobol_strat]

            [manual_strat]
            generator = ManualGenerator

            [ManualGenerator]
            points = [[0.5], [0.2]]

            [sobol_strat]
            min_asks = 2
            generator = SobolGenerator
            min_total_outcome_occurrences = 0

        """
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": exp_config},
        }
        ask_request = {"type": "ask", "message": ""}
        tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5]}, "outcome": 1},
        }
        exit_request = {"message": "", "type": "exit"}

        self.s.handle_request(setup_request)
        while not self.s.strat.finished:
            self.s.handle_request(ask_request)
            self.s.handle_request(tell_request)

        self.s.handle_request(exit_request)

        socket = server.sockets.PySocket(port=0)
        serv = server.AEPsychServer(socket=socket, database_path=self.db_path)
        exp_ids = [rec.unique_id for rec in serv.db.get_master_records()]

        serv.replay(exp_ids[-1], skip_computations=True)
        strat = serv._strats[-1]

        self.assertTrue(strat._strat_idx == 1)
        self.assertTrue(strat.finished)
        self.assertTrue(strat.x.shape[0] == 4)

    def test_string_parameter(self):
        string_config = """
            [common]
            parnames = [x, y, z]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat, opt_strat]

            [x]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [y]
            par_type = fixed
            value = blue

            [z]
            par_type = integer
            lower_bound = 0
            upper_bound = 100

            [init_strat]
            min_asks = 2
            generator = SobolGenerator
            min_total_outcome_occurrences = 0

            [opt_strat]
            min_asks = 2
            generator = OptimizeAcqfGenerator
            model = GPClassificationModel
            min_total_outcome_occurrences = 0

            [OptimizeAcqfGenerator]
            acqf = MCLevelSetEstimation
        """
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": string_config},
        }
        ask_request = {"type": "ask", "message": ""}
        tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5], "y": ["blue"], "z": [50]}, "outcome": 1},
        }
        self.s.handle_request(setup_request)
        while not self.s.strat.finished:
            response = self.s.handle_request(ask_request)
            self.assertTrue(response["config"]["y"][0] == "blue")
            self.s.handle_request(tell_request)

        self.assertTrue(torch.all(torch.tensor([0, 0, 0]) == self.s.strat.lb))
        self.assertTrue(torch.all(torch.tensor([1, 1, 100]) == self.s.strat.ub))

    def test_metadata(self):
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

        master_record = self.s.db.get_master_records()[-1]
        extra_metadata = json.loads(master_record.extra_metadata)

        self.assertTrue(master_record.experiment_name == "test experiment")
        self.assertTrue(
            master_record.experiment_description
            == "dummy experiment to test the server"
        )
        self.assertTrue(master_record.experiment_id == "e1")
        self.assertTrue(master_record.participant_id == "101")
        self.assertTrue(extra_metadata["extra"] == "data that is arbitrary")
        self.assertTrue("experiment_id" not in extra_metadata)

    def test_extension_server(self):
        extension_path = Path(__file__).parent.parent.parent
        extension_path = extension_path / "extensions_example" / "new_objects.py"

        config_str = f"""
            [common]
            parnames = [signal1]
            outcome_types = [binary]
            stimuli_per_trial = 1
            strategy_names = [opt_strat]
            extensions = [{extension_path}]

            [signal1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [opt_strat]
            model = GPClassificationModel
            generator = OnesGenerator
            min_asks = 1
        """
        setup_request = {
            "type": "setup",
            "message": {"config_str": config_str},
        }

        with self.assertLogs(level=logging.INFO) as logs:
            self.s.handle_request(setup_request)
            outputs = ";".join(logs.output)
            self.assertTrue(str(extension_path) in outputs)

        strat = self.s.strat
        one = strat.gen()

        self.assertTrue(one == 1)
        self.assertTrue(strat.generator._base_obj.__class__.__name__ == "OnesGenerator")

        self.s.extensions.unload()

    def test_read_only_mode(self):
        """Test that read-only mode creates a temporary copy of the database and doesn't modify the original."""
        # Create a server with a database and add some data to it
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
        self.s.handle_request(ask_request)
        self.s.handle_request(tell_request)

        # Get the database state before read-only operations
        original_records_count = len(self.s.db.get_master_records())

        # Create a second server with the same database path but in read-only mode
        socket = server.sockets.PySocket(port=0)
        read_only_server = server.AEPsychServer(
            socket=socket, database_path=self.db_path, read_only=True
        )

        # Verify that the read-only server has a _temp_dir attribute
        self.assertTrue(hasattr(read_only_server.db, "_temp_dir"))

        # Verify that the database path being used is different from the original
        self.assertNotEqual(read_only_server.db._full_db_path.as_posix(), self.db_path)

        # Verify that the temporary database contains the same data as the original
        self.assertEqual(
            len(read_only_server.db.get_master_records()), original_records_count
        )

        # Perform operations on the read-only server
        read_only_server.handle_request(setup_request)
        read_only_server.handle_request(ask_request)
        read_only_server.handle_request(tell_request)

        # Verify that the read-only server's database has been modified
        self.assertEqual(
            len(read_only_server.db.get_master_records()),
            original_records_count + 1,
        )

        # Store the temp_dir path for later verification
        temp_dir_path = read_only_server.db._temp_dir.name

        # Clean up the read-only server
        read_only_server.cleanup()

        # Create a new server to check the original database
        socket = server.sockets.PySocket(port=0)
        check_server = server.AEPsychServer(socket=socket, database_path=self.db_path)

        # Verify that the original database was not modified
        self.assertEqual(
            len(check_server.db.get_master_records()), original_records_count
        )

        # Verify that the temporary directory is gone
        self.assertFalse(os.path.exists(temp_dir_path))

    def test_read_only_mode_file_not_found(self):
        """Test that read-only mode raises FileNotFoundError when the database file doesn't exist."""
        # Create a path to a non-existent database file
        non_existent_db_path = os.path.join(self.db_path + "_non_existent")

        # Verify that the file doesn't exist
        self.assertFalse(os.path.exists(non_existent_db_path))

        # Try to create a server with the non-existent database path in read-only mode
        socket = server.sockets.PySocket(port=0)
        with self.assertRaises(FileNotFoundError):
            server.AEPsychServer(
                socket=socket, database_path=non_existent_db_path, read_only=True
            )


class BackgroundServerTestCase(unittest.IsolatedAsyncioTestCase):
    """Test case for testing server behavior with a running server in a background thread."""

    async def asyncSetUp(self):
        """Set up a server instance running in a background thread."""
        # Multithread socket doesn't play nice, so we ignore these warnings
        warnings.filterwarnings(
            "ignore", message="unclosed <socket.socket ", category=ResourceWarning
        )
        # Create a server instance with port
        self.ip = "localhost"
        # 0 is a special port number that tells the OS to assign an available port
        self.server_socket = server.sockets.PySocket(ip=self.ip, port=0)
        self.port = self.server_socket.socket.getsockname()[1]
        self.database_path = "./{}_test_server.db".format(str(uuid.uuid4().hex))
        self.server_instance = server.AEPsychServer(
            socket=self.server_socket, database_path=self.database_path
        )

        # Start the server in a separate thread
        self.server_thread = threading.Thread(
            target=self.server_instance.serve, daemon=True
        )
        self.server_thread.start()

        # Give the server time to start up
        time.sleep(1)

        # Create a client socket and connect to the server
        # Explicitly create an IPv4 socket
        self.reader, self.writer = await asyncio.open_connection(self.ip, self.port)
        self.reader_lock = asyncio.Lock()

    async def asyncTearDown(self):
        """Clean up resources after the test."""
        self.writer.close()
        await self.writer.wait_closed()

        # Clean up the server
        self.server_instance.cleanup()
        self.server_instance.exit_server_loop = True
        self.server_thread.join(timeout=5)

        # Delete the database
        try:
            self.server_instance.db.delete_db()
        except PermissionError as e:
            print("Failed to delete database: ", e)

    async def mock_client(self, request):
        self.writer.write(json.dumps(request).encode())
        await self.writer.drain()
        async with self.reader_lock:
            response = await self.reader.read(1024 * 512)
        return json.loads(response.decode())

    async def test_exception_json_response(self):
        """Test that the server returns a JSON response when an exception occurs."""
        # First send a valid setup request
        setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": dummy_config},
        }
        _ = await self.mock_client(setup_request)

        # Now send a malformed tell request that will cause an exception
        # Missing the required 'outcome' field in the message
        malformed_tell_request = {
            "type": "tell",
            "message": {"config": {"x": [0.5]}},  # Missing 'outcome' field
        }

        with self.assertLogs() as log:
            response = await self.mock_client(malformed_tell_request)

        self.assertIn(
            "tell() missing 1 required positional argument: 'outcome'", log.output[-1]
        )

        # Verify that the response contains the server_error key
        self.assertIn("server_error", response)
        # Verify that the original message is included in the response
        self.assertEqual(response["message"], malformed_tell_request)

    @patch.dict("aepsych.server.message_handlers.MESSAGE_MAP", {"wait": handle_wait})
    async def test_queue(self):
        """Test that queue is being handled correctly"""
        terminate_wait_request = {"type": "wait", "message": {"terminate": True}}

        results = await asyncio.gather(self.mock_client(terminate_wait_request))
        self.assertEqual(len(self.server_instance.queue), 0)
        self.assertEqual(results[0]["queue_size"], 0)

        self.server_thread.join(timeout=5)
        self.assertFalse(self.server_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
