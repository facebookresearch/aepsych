#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from aepsych.generators import SobolGenerator
from aepsych.strategy import Strategy


class TestStrategyGPU(unittest.TestCase):
    def test_gpu_no_model_generator_warn(self):
        with self.assertLogs() as log:
            Strategy(
                lb=[0.0],
                ub=[1.0],
                stimuli_per_trial=1,
                outcome_types=["binary"],
                min_asks=1,
                generator=SobolGenerator(lb=[0], ub=[1]),
                use_gpu_generating=True,
            )

        self.assertIn(
            "GPU requested for generator SobolGenerator but this generator has no model to move to GPU",
            log.output[0],
        )


if __name__ == "__main__":
    unittest.main()
