#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from aepsych.generators import OptimizeAcqfGenerator
from aepsych.models.base import ModelProtocol
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption


class PairwiseOptimizeAcqfGenerator(OptimizeAcqfGenerator):
    def _instantiate_acquisition_fn(self, model, train_x):
        if self.acqf == AnalyticExpectedUtilityOfBestOption:
            return self.acqf(pref_model=model)
        else:
            return super()._instantiate_acquisition_fn(model, train_x)

    def gen(self, num_points: int, model: ModelProtocol, **gen_options) -> np.ndarray:

        qbatch_points = super().gen(
            num_points=num_points * 2, model=model, **gen_options
        )

        # output of super() is (q, dim) but the contract is (num_points, dim, 2)
        # so we need to split q into q and pairs and then move the pair dim to the end
        return qbatch_points.reshape(num_points, 2, -1).swapaxes(-1, -2)
