#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
These tests check that the server can handle different experiments
(multi/single stimuli, multi/single outcome). They ensure that the
data is correctly stored in the database tables (raw, param, and outcome).
It also checks that the experiment table is correctly populated
(generate_experiment_table method).
"""

import logging
import unittest
import uuid
from itertools import product

import aepsych.server as server
import aepsych.utils_logging as utils_logging
from parameterized import parameterized

params = {
    "singleStimuli": {
        "x1": [0.1, 0.2, 0.3, 1, 2, 3, 4],
        "x2": [4, 0.1, 3, 0.2, 2, 1, 0.3],
    },
    "multiStimuli": {
        "x1": [[0.1, 0.2], [0.3, 1], [2, 3], [4, 0.1], [0.2, 2], [1, 0.3], [0.3, 0.1]],
        "x2": [[4, 0.1], [3, 0.2], [2, 1], [0.3, 0.2], [2, 0.3], [1, 0.1], [0.3, 4]],
    },
}

outcomes = {
    "singleOutcome": [1, -1, 0.1, 0, -0.1, 0, 0],
    "multiOutcome": [
        [[1], [0]],
        [[-1], [0]],
        [[0.1], [0]],
        [[0], [0]],
        [[-0.1], [0]],
        [[0], [0]],
        [[0], [0]],
    ],
}

all_tests = list(product(params, outcomes))


class IntegrationTestCase(unittest.TestCase):
    def setUp(self):
        # setup logger
        server.logger = utils_logging.getLogger(logging.DEBUG, "logs")

        # random port
        socket = server.sockets.PySocket(port=0)

        # random datebase path name without dashes
        database_path = "./{}.db".format(str(uuid.uuid4().hex))
        self.s = server.AEPsychServer(socket=socket, database_path=database_path)

        # Server messages
        self.setup_request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": None},
        }
        self.ask_request = {"type": "ask", "message": ""}
        self.tell_request = {
            "type": "tell",
            "message": {"config": {}, "outcome": 0},
            "extra_info": {},
        }

    def tearDown(self):
        self.s.cleanup()

        # cleanup the db
        if self.s.db is not None:
            self.s.db.delete_db()

    def get_tell(self, x1, x2, outcome):
        self.tell_request["message"]["config"]["x1"] = x1
        self.tell_request["message"]["config"]["x2"] = x2
        self.tell_request["message"]["outcome"] = outcome
        self.tell_request["extra_info"]["e1"] = 1
        self.tell_request["extra_info"]["e2"] = 2

    def check_params(self, param_type, x1, x2):
        if param_type == "multiStimuli":
            x1_stimuli0_saved = (
                self.s.db.get_engine()
                .execute("SELECT x1_stimuli0 FROM experiment_table")
                .fetchall()
            )
            x1_stimuli1_saved = (
                self.s.db.get_engine()
                .execute("SELECT x1_stimuli1 FROM experiment_table")
                .fetchall()
            )
            x1_stimuli0_saved = [
                float(item) for sublist in x1_stimuli0_saved for item in sublist
            ]
            x1_stimuli1_saved = [
                float(item) for sublist in x1_stimuli1_saved for item in sublist
            ]

            # Reshape
            x1_saved = []
            for i in range(len(x1_stimuli0_saved)):
                x1_saved.append([x1_stimuli0_saved[i], x1_stimuli1_saved[i]])
            self.assertEqual(x1_saved, x1)

            x2_stimuli0_saved = (
                self.s.db.get_engine()
                .execute("SELECT x2_stimuli0 FROM experiment_table")
                .fetchall()
            )
            x2_stimuli1_saved = (
                self.s.db.get_engine()
                .execute("SELECT x2_stimuli1 FROM experiment_table")
                .fetchall()
            )
            x2_stimuli0_saved = [
                float(item) for sublist in x2_stimuli0_saved for item in sublist
            ]
            x2_stimuli1_saved = [
                float(item) for sublist in x2_stimuli1_saved for item in sublist
            ]

            # Reshape
            x2_saved = []
            for i in range(len(x2_stimuli0_saved)):
                x2_saved.append([x2_stimuli0_saved[i], x2_stimuli1_saved[i]])
            self.assertEqual(x2_saved, x2)
        elif param_type == "singleStimuli":
            x1_saved = (
                self.s.db.get_engine()
                .execute("SELECT x1 FROM experiment_table")
                .fetchall()
            )
            x1_saved = [float(item) for sublist in x1_saved for item in sublist]
            self.assertTrue(x1_saved == x1)

            x2_saved = (
                self.s.db.get_engine()
                .execute("SELECT x2 FROM experiment_table")
                .fetchall()
            )
            x2_saved = [float(item) for sublist in x2_saved for item in sublist]
            self.assertTrue(x2_saved == x2)

    def check_outcome(self, outcome_type, outcome):
        if outcome_type == "multiOutcome":
            outcome0_saved = (
                self.s.db.get_engine()
                .execute("SELECT outcome_0 FROM experiment_table")
                .fetchall()
            )
            outcome1_saved = (
                self.s.db.get_engine()
                .execute("SELECT outcome_1 FROM experiment_table")
                .fetchall()
            )
            outcome0_saved = [item for sublist in outcome0_saved for item in sublist]
            outcome1_saved = [item for sublist in outcome1_saved for item in sublist]
            outcome_saved = []
            for i in range(len(outcome0_saved)):
                outcome_saved.append([[outcome0_saved[i]], [outcome1_saved[i]]])
            self.assertEqual(outcome_saved, outcome)
        elif outcome_type == "singleOutcome":
            outcome_saved = (
                self.s.db.get_engine()
                .execute("SELECT outcome FROM experiment_table")
                .fetchall()
            )
            outcome_saved = [item for sublist in outcome_saved for item in sublist]
            self.assertTrue(outcome_saved == outcome)

    @parameterized.expand(all_tests)
    def test_experiment(self, param_type, outcome_type):
        x1 = params[param_type]["x1"]
        x2 = params[param_type]["x2"]
        outcome = outcomes[outcome_type]

        with open(f"tests/configs/{param_type}" + ".ini", "r") as f:
            dummy_config = f.read()

        self.setup_request["message"]["config_str"] = dummy_config

        self.s.versioned_handler(self.setup_request)

        i = 0
        while not self.s.strat.finished:
            self.s.unversioned_handler(self.ask_request)
            self.get_tell(x1[i], x2[i], outcome[i])
            i = i + 1
            self.s.unversioned_handler(self.tell_request)

        # Experiment id
        exp_id = self.s.db.get_master_records()[0].experiment_id

        # Create table with experiment data
        self.s.generate_experiment_table(exp_id, return_df=True)

        # Check that table exists
        self.assertTrue("experiment_table" in self.s.db.get_engine().table_names())

        # Check that parameter and outcomes values are correct
        self.check_outcome(outcome_type, outcome)
        self.check_params(param_type, x1, x2)


if __name__ == "__main__":
    unittest.main()
