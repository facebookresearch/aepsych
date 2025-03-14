#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest
from unittest.mock import MagicMock

from ..test_server import BaseServerTestCase, dummy_config


class MessageHandlerTellTests(BaseServerTestCase):
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

        self.s.handle_request(setup_request)
        msg = self.s.handle_request(tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 1)
        self.assertEqual(len(self.s.strat.x), 1)
        self.assertEqual(msg["trials_recorded"], 1)
        self.assertEqual(msg["model_data_added"], 1)

        tell_request["message"]["model_data"] = False
        msg = self.s.handle_request(tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 2)
        self.assertEqual(len(self.s.strat.x), 1)
        self.assertEqual(msg["trials_recorded"], 1)
        self.assertEqual(msg["model_data_added"], 0)

    def test_batch_tell(self):
        setup_request = {
            "type": "setup",
            "message": {"config_str": dummy_config},
        }

        batch_tell_request = {
            "type": "tell",
            "message": {"config": {"x": [[0.5], [1.0], [0.0]]}, "outcome": [1, 0, 1]},
        }

        self.s.db.record_message = MagicMock()

        self.s.handle_request(setup_request)
        msg = self.s.handle_request(batch_tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 1)
        self.assertEqual(len(self.s.strat.x), 3)
        self.assertEqual(msg["trials_recorded"], 3)
        self.assertEqual(msg["model_data_added"], 3)

        self.s.handle_request(batch_tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 2)
        self.assertEqual(len(self.s.strat.x), 6)

        batch_tell_request["message"]["model_data"] = False
        msg = self.s.handle_request(batch_tell_request)
        self.assertEqual(self.s.db.record_message.call_count, 3)
        self.assertEqual(len(self.s.strat.x), 6)
        self.assertEqual(msg["trials_recorded"], 3)
        self.assertEqual(msg["model_data_added"], 0)

    def test_tell_extra_data(self):
        setup_request = {
            "type": "setup",
            "message": {"config_str": dummy_config},
        }

        tell_request = {
            "type": "tell",
            "message": {
                "config": {"x": [0.5]},
                "outcome": 1,
                "extra": "data",
                "additional": 1,
            },
        }

        self.s.db.record_message = MagicMock()

        self.s.handle_request(setup_request)
        self.s.handle_request(tell_request)

        raw_row = self.s.db.get_raw_for(1)[0]
        extra_data = json.loads(raw_row.extra_data)
        self.assertTrue(extra_data["extra"] == "data")
        self.assertTrue(extra_data["additional"] == 1)


class MultiOutcomeTellTests(BaseServerTestCase):
    def setUp(self):
        super().setUp()
        config_str = """
            [common]
            parnames = [x]
            stimuli_per_trial = 1
            outcome_types = [binary, continuous]
            outcome_names = [binOut, contOut]
            strategy_names = [init_strat, opt_strat]

            [x]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            min_asks = 2
            generator = SobolGenerator
            min_total_outcome_occurrences = 0

            [opt_strat]
            min_asks = 2
            generator = IndependentOptimizeAcqfGenerator
            model = IndependentGPsModel
            min_total_outcome_occurrences = 0

            [IndependentOptimizeAcqfGenerator]
            generators = [binGen, contGen]

            [binGen]
            class = OptimizeAcqfGenerator
            acqf = MCLevelSetEstimation

            [contGen]
            class = OptimizeAcqfGenerator
            acqf = qLogNoisyExpectedImprovement

            [IndependentGPsModel]
            models = [binModel, contModel]

            [binModel]
            class = GPClassificationModel
            inducing_size = 10

            [contModel]
            class = GPRegressionModel
        """

        setup_request = {
            "type": "setup",
            "message": {"config_str": config_str},
        }

        self.s.db.record_message = MagicMock()

        self.s.handle_request(setup_request)

    def test_single_trial(self):
        request = {
            "type": "tell",
            "message": {
                "config": {"x": [0.5]},
                "outcome": {"contOut": 0.5, "binOut": 1},
            },
        }

        response = self.s.handle_request(request)
        response = self.s.handle_request(request)
        response = self.s.handle_request(request)

        self.assertEqual(response["trials_recorded"], 1)
        self.assertEqual(self.s.strat.y.shape, (3, 2))
        self.assertEqual(self.s.strat.y[0, 0], 1.0)
        self.assertEqual(self.s.strat.y[0, 1], 0.5)

    def test_list_mixed_trial(self):
        request = {
            "type": "tell",
            "message": {
                "config": {"x": [0.5]},
                "outcome": {"contOut": [0.5], "binOut": "1"},
            },
        }

        response = self.s.handle_request(request)
        response = self.s.handle_request(request)
        response = self.s.handle_request(request)

        self.assertEqual(response["trials_recorded"], 1)
        self.assertEqual(self.s.strat.y.shape, (3, 2))
        self.assertEqual(self.s.strat.y[0, 0], 1.0)
        self.assertEqual(self.s.strat.y[0, 1], 0.5)

    def test_multi_trial(self):
        request = {
            "type": "tell",
            "message": {
                "config": {"x": [[0.5], [0.75]]},
                "outcome": {"contOut": [0.5, 0.5], "binOut": [1, 0]},
            },
        }

        response = self.s.handle_request(request)
        response = self.s.handle_request(request)

        self.assertEqual(response["trials_recorded"], 2)
        self.assertEqual(self.s.strat.y.shape, (4, 2))
        self.assertEqual(self.s.strat.y[0, 0], 1.0)
        self.assertEqual(self.s.strat.y[0, 1], 0.5)

    def test_wrong_keys(self):
        # Test with missing outcome key
        request_missing_key = {
            "type": "tell",
            "message": {
                "config": {"x": [0.5]},
                "outcome": {"contOut": 0.5},  # Missing 'binOut'
            },
        }

        with self.assertRaises(KeyError):
            self.s.handle_request(request_missing_key)

        # Test with extra outcome key
        request_extra_key = {
            "type": "tell",
            "message": {
                "config": {"x": [0.5]},
                "outcome": {
                    "contOut": 0.5,
                    "binOut": 1,
                    "extraOut": 0.1,
                },  # Extra 'extraOut'
            },
        }

        with self.assertRaises(KeyError):
            self.s.handle_request(request_extra_key)


if __name__ == "__main__":
    unittest.main()
