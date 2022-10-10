#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest
import uuid
from unittest.mock import MagicMock, patch

import torch
from aepsych.acquisition import MCPosteriorVariance
from aepsych.generators import OptimizeAcqfGenerator, SobolGenerator
from aepsych.models import GPClassificationModel
from aepsych.server import AEPsychServer
from aepsych_client import AEPsychClient
from torch import tensor


class MockStrategy:
    def gen(self, num_points):
        self._count = self._count + num_points
        return torch.tensor([[0.0]])


class RemoteServerTestCase(unittest.TestCase):
    def setUp(self):
        database_path = "./{}.db".format(str(uuid.uuid4().hex))
        self.s = AEPsychServer(database_path=database_path)
        self.client = AEPsychClient(connect=False)
        self.client._send_recv = MagicMock(
            wraps=lambda x: json.dumps(self.s.handle_request(x))
        )

    def tearDown(self):
        self.s.cleanup()

        # cleanup the db
        if self.s.db is not None:
            self.s.db.delete_db()

    @patch(
        "aepsych.strategy.Strategy.gen",
        new=MockStrategy.gen,
    )
    def test_client(self):
        config_str = """
        [common]
        lb = [0]
        ub = [1]
        parnames = [x]
        stimuli_per_trial = 1
        outcome_types = [binary]
        strategy_names = [init_strat, opt_strat]
        acqf = MCPosteriorVariance
        model = GPClassificationModel

        [init_strat]
        min_asks = 1
        generator = SobolGenerator
        min_total_outcome_occurrences = 0

        [opt_strat]
        min_asks = 1
        generator = OptimizeAcqfGenerator
        min_total_outcome_occurrences = 0
        """

        self.client.configure(config_str=config_str, config_name="first_config")
        self.assertEqual(self.s.strat_id, 0)
        self.assertEqual(self.s.strat.strat_list[0].min_asks, 1)
        self.assertEqual(self.s.strat.strat_list[1].min_asks, 1)
        self.assertIsInstance(self.s.strat.strat_list[0].generator, SobolGenerator)
        self.assertIsInstance(
            self.s.strat.strat_list[1].generator, OptimizeAcqfGenerator
        )
        self.assertIsInstance(self.s.strat.strat_list[1].model, GPClassificationModel)
        self.assertEqual(self.s.strat.strat_list[1].generator.acqf, MCPosteriorVariance)

        response = self.client.ask()
        self.assertSetEqual(set(response["config"].keys()), {"x"})
        self.assertEqual(len(response["config"]["x"]), 1)
        self.assertTrue(0 <= response["config"]["x"][0] <= 1)
        self.assertFalse(response["is_finished"])
        self.assertEqual(self.s.strat._count, 1)

        self.client.tell(config={"x": [0]}, outcome=1)
        self.assertEqual(self.s._strats[0].x, tensor([[0.0]]))
        self.assertEqual(self.s._strats[0].y, tensor([[1.0]]))

        self.client.tell(config={"x": [0]}, outcome=1, model_data=False)
        self.assertEqual(self.s._strats[0].x, tensor([[0.0]]))
        self.assertEqual(self.s._strats[0].y, tensor([[1.0]]))

        response = self.client.ask()
        self.assertTrue(response["is_finished"])

        self.client.configure(config_str=config_str, config_name="second_config")
        self.assertEqual(self.s.strat._count, 0)
        self.assertEqual(self.s.strat_id, 1)

        self.client.resume(config_name="first_config")
        self.assertEqual(self.s.strat_id, 0)

        self.client.resume(config_name="second_config")
        self.assertEqual(self.s.strat_id, 1)

        self.client.finalize()


class LocalServerTestCase(RemoteServerTestCase):
    def setUp(self):
        database_path = "./{}.db".format(str(uuid.uuid4().hex))
        self.s = AEPsychServer(database_path=database_path)
        self.client = AEPsychClient(server=self.s)

    def test_warns_ignored_args(self):
        with self.assertWarns(UserWarning):
            AEPsychClient(ip="0.0.0.0", port=5555, server=self.s)


if __name__ == "__main__":
    unittest.main()
