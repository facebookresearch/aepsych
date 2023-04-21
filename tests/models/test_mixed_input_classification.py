#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import numpy as np
import torch
from aepsych.generators import SobolGenerator
from aepsych.models import MixedInputGPClassificationModel
from scipy.stats import bernoulli, norm

from ..common import new_novel_det_channels

# run on single threads to keep us from deadlocking weirdly in CI
if "CI" in os.environ or "SANDCASTLE" in os.environ:
    torch.set_num_threads(1)


class MixedInputGPClassificationTest(unittest.TestCase):
    def test_mixedinput_classification_smoke(self):
        np.random.seed(0)
        torch.manual_seed(0)
        n_train = 20
        n_test = 10
        generator = SobolGenerator(lb=[-1], ub=[1], dim=1)
        x = generator.gen(n_train)
        channel = np.random.choice(2, n_train)
        f = new_novel_det_channels(x.squeeze(), channel)
        y = bernoulli.rvs(norm.cdf(f))
        xtest = generator.gen(n_test)

        f1_true = new_novel_det_channels(xtest.squeeze(), 0)
        f2_true = new_novel_det_channels(xtest.squeeze(), 1)

        model = MixedInputGPClassificationModel(
            lb=[-1], ub=[1], discrete_param_levels=[2], discrete_param_ranks=[2]
        )

        model.fit(
            torch.Tensor(x),
            torch.Tensor(channel),
            torch.Tensor(y),
            optimizer_kwargs={"options": {"maxfun": 100}},
        )

        p1_pred, _ = model.predict(xtest, torch.Tensor([0]), probability_space=True)
        num_mismatches = (
            (p1_pred > 0.5).numpy() != (norm.cdf(f1_true.squeeze()) > 0.5)
        ).sum()
        self.assertLessEqual(num_mismatches, 1)

        p2_pred, _ = model.predict(xtest, torch.Tensor([1]), probability_space=True)
        num_mismatches = (
            (p2_pred > 0.5).numpy() != (norm.cdf(f2_true.squeeze()) > 0.5)
        ).sum()
        self.assertLessEqual(num_mismatches, 1)


if __name__ == "__main__":
    unittest.main()
