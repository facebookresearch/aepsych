#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from aepsych.config import Config
from aepsych.strategy import SequentialStrategy


class ConfigTestCase(unittest.TestCase):
    def test_model_gpu(self):
        config_str = """
            [common]
            parnames = [par1]
            strategy_names = [gpu_strategy]
            outcome_types = [binary]
            stimuli_per_trial = 1

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1
            
            [gpu_strategy]
            model = GPClassificationModel
            generator = SobolGenerator
            
            [GPClassificationModel]
            use_gpu = True
        """
        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)

        self.assertTrue(strat.model.device.type == "cuda")

        strat.add_data(torch.tensor([0.5]), torch.tensor([1]))
        strat.fit()

        self.assertTrue(strat.model.device.type == "cuda")


if __name__ == "__main__":
    unittest.main()
