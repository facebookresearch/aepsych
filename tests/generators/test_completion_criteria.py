#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from aepsych.config import Config
from aepsych.generators.completion_criterion import (
    MinAsks,
    MinTotalOutcomeOccurrences,
    MinTotalTells,
)
from aepsych.strategy import AEPsychStrategy


class CompletionCriteriaTestCase(unittest.TestCase):
    def setUp(self):
        config_str = """
        [common]
        use_ax = True
        stimuli_per_trial = 1
        outcome_types = [binary]
        parnames = [x]
        lb = [0]
        ub = [1]
        strategy_names = [test_strat]

        [test_strat]
        generator = SobolGenerator
        """
        config = Config(config_str=config_str)
        self.strat = AEPsychStrategy.from_config(config)

    def test_min_asks(self):
        config_str = """
        [test_strat]
        min_asks = 2
        """
        config = Config(config_str=config_str)
        criterion = MinAsks.from_config(config, "test_strat")
        self.assertEqual(criterion.threshold, 2)

        self.assertFalse(criterion.is_met(self.strat.experiment))

        self.strat.complete_new_trial({"x": 0.0}, 0.0)
        self.assertFalse(criterion.is_met(self.strat.experiment))

        self.strat.complete_new_trial({"x": 1.0}, 0.0)
        self.assertFalse(criterion.is_met(self.strat.experiment))

        self.strat.gen()
        self.assertFalse(criterion.is_met(self.strat.experiment))

        self.strat.gen()
        self.assertTrue(criterion.is_met(self.strat.experiment))

    def test_min_total_tells(self):
        config_str = """
        [test_strat]
        min_total_tells = 2
        """
        config = Config(config_str=config_str)
        criterion = MinTotalTells.from_config(config, "test_strat")
        self.assertEqual(criterion.threshold, 2)

        self.assertFalse(criterion.is_met(self.strat.experiment))

        self.strat.gen()
        self.assertFalse(criterion.is_met(self.strat.experiment))

        self.strat.gen()
        self.assertFalse(criterion.is_met(self.strat.experiment))

        self.strat.complete_new_trial({"x": 0.0}, 0.0)
        self.assertFalse(criterion.is_met(self.strat.experiment))

        self.strat.complete_new_trial({"x": 1.0}, 0.0)
        self.assertTrue(criterion.is_met(self.strat.experiment))

    def test_min_total_outcome_occurences(self):
        config_str = """
        [common]
        outcome_types = [binary]
        min_total_outcome_occurrences = 2
        """
        config = Config(config_str=config_str)
        criterion = MinTotalOutcomeOccurrences.from_config(config, "test_strat")
        self.assertEqual(criterion.threshold, 2)

        self.strat.complete_new_trial({"x": 0.0}, 0.0)
        self.assertFalse(criterion.is_met(self.strat.experiment))

        self.strat.complete_new_trial({"x": 1.0}, 0.0)
        self.assertFalse(criterion.is_met(self.strat.experiment))

        self.strat.complete_new_trial({"x": 0.0}, 1.0)
        self.assertFalse(criterion.is_met(self.strat.experiment))

        self.strat.complete_new_trial({"x": 1.0}, 1.0)
        self.assertTrue(criterion.is_met(self.strat.experiment))


if __name__ == "__main__":
    unittest.main()
