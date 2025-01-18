#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from aepsych.acquisition.monotonic_rejection import MonotonicMCLSE
from aepsych.generators import OptimizeAcqfGenerator, SobolGenerator
from aepsych.models.gp_classification import GPClassificationModel
from aepsych.strategy import Strategy


class TestStrategyGPU(unittest.TestCase):
    def test_gpu_no_model_generator_warn(self):
        with self.assertWarns(UserWarning):
            Strategy(
                lb=[0.0],
                ub=[1.0],
                stimuli_per_trial=1,
                outcome_types=["binary"],
                min_asks=1,
                generator=SobolGenerator(lb=[0], ub=[1]),
                use_gpu_generating=True,
            )

    def test_no_gpu_acqf(self):
        with self.assertWarns(UserWarning):
            Strategy(
                lb=[0.0],
                ub=[1.0],
                stimuli_per_trial=1,
                outcome_types=["binary"],
                min_asks=1,
                model=GPClassificationModel(
                    dim=1,
                ),
                generator=OptimizeAcqfGenerator(acqf=MonotonicMCLSE, lb=[0], ub=[1]),
                use_gpu_generating=True,
            )


if __name__ == "__main__":
    unittest.main()
