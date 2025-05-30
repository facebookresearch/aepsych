#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch

# run on single threads to keep us from deadlocking weirdly in CI
if "CI" in os.environ or "SANDCASTLE" in os.environ:
    torch.set_num_threads(1)

import numpy as np
from aepsych.config import Config
from aepsych.transforms import ParameterTransformedModel
from sklearn.datasets import make_classification


class MonotonicProjectionGPtest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        torch.manual_seed(1)
        X, y = make_classification(
            n_samples=100,
            n_features=3,
            n_redundant=0,
            n_informative=3,
            random_state=1,
            n_clusters_per_class=1,
        )
        self.X, self.y = torch.Tensor(X), torch.Tensor(y)

    def test_posterior(self):
        X, y = self.X, self.y
        config_str = """
        [common]
        parnames = [x, y, z]
        lb = [-4, -4, -4]
        ub = [4, 4, 4]
        stimuli_per_trial = 1
        outcome_types = [binary]

        strategy_names = [init_strat]

        [init_strat]
        generator = OptimizeAcqfGenerator
        model = MonotonicProjectionGP

        [MonotonicProjectionGP]
        monotonic_dims = [0, 1]
        inducing_size=100

        [DefaultMeanCovarFactory]
        lengthscale_prior = gamma
        fixed_kernel_amplitude = False
        """
        config = Config(config_str=config_str)
        model = ParameterTransformedModel.from_config(config, "MonotonicProjectionGP")
        model.cuda()
        model.fit(X, y)

        # Check that it is monotonic in both dims
        for i in range(2):
            Xtest = torch.zeros(3, 3)
            Xtest[:, i] = torch.tensor([-1, 0, 1])
            post = model.posterior(Xtest)
            mu = post.mean.squeeze()
            self.assertTrue(mu[0] <= mu[1] and mu[1] <= mu[2])

        # Check that min_f_val is respected
        config_str = """
        [common]
        parnames = [x, y, z]
        lb = [-4, -4, -4]
        ub = [4, 4, 4]
        stimuli_per_trial = 1
        outcome_types = [binary]

        strategy_names = [init_strat]

        [init_strat]
        generator = OptimizeAcqfGenerator
        model = MonotonicProjectionGP

        [MonotonicProjectionGP]
        monotonic_dims = [0]
        inducing_size=10
        min_f_val = 5.0

        [DefaultMeanCovarFactory]
        lengthscale_prior = gamma
        fixed_kernel_amplitude = False
        """
        config = Config(config_str=config_str)
        model = ParameterTransformedModel.from_config(config, "MonotonicProjectionGP")
        post = model.posterior(Xtest)
        mu = post.mean.squeeze()
        self.assertTrue(mu.min().item() >= 4.9)
        # And in samples
        samps = model.sample(Xtest, num_samples=10)
        self.assertTrue(samps.min().item() >= 4.9)


if __name__ == "__main__":
    unittest.main()
