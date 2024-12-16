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
import numpy.testing as npt
from aepsych.models import GPClassificationModel
from scipy.stats import norm
from sklearn.datasets import make_classification


def f_1d(x, mu=0):
    """
    latent is just a gaussian bump at mu
    """
    return np.exp(-((x - mu) ** 2))


def f_2d(x):
    """
    a gaussian bump at 0 , 0
    """
    return np.exp(-np.linalg.norm(x, axis=-1))


def new_novel_det_params(freq, scale_factor=1.0):
    """Get the loc and scale params for 2D synthetic novel_det(frequency) function
        Keyword arguments:
    freq -- 1D array of frequencies whose thresholds to return
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    target -- target threshold
    """
    locs = 0.66 * np.power(0.8 * freq * (0.2 * freq - 1), 2) + 0.05
    scale = 2 * locs / (3 * scale_factor)
    loc = -1 + 2 * locs
    return loc, scale


def target_new_novel_det(freq, scale_factor=1.0, target=0.75):
    """Get the target (i.e. threshold) for 2D synthetic novel_det(frequency) function
        Keyword arguments:
    freq -- 1D array of frequencies whose thresholds to return
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    target -- target threshold
    """
    locs, scale = new_novel_det_params(freq, scale_factor)
    return norm.ppf(target, loc=locs, scale=scale)


def new_novel_det(x, scale_factor=1.0):
    """Get the cdf for 2D synthetic novel_det(frequency) function
        Keyword arguments:
    x -- array of shape (n,2) of locations to sample;
         x[...,0] is frequency from -1 to 1; x[...,1] is intensity from -1 to 1
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    """
    freq = x[..., 0]
    locs, scale = new_novel_det_params(freq, scale_factor)
    return (x[..., 1] - locs) / scale


def cdf_new_novel_det(x, scale_factor=1.0):
    """Get the cdf for 2D synthetic novel_det(frequency) function
        Keyword arguments:
    x -- array of shape (n,2) of locations to sample;
         x[...,0] is frequency from -1 to 1; x[...,1] is intensity from -1 to 1
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    """
    return norm.cdf(new_novel_det(x, scale_factor))


class GPClassificationSmoketest(unittest.TestCase):
    """
    Super basic smoke test to make sure we know if we broke the underlying model
    for single-probit  ("1AFC") model
    """

    def setUp(self):
        np.random.seed(1)
        torch.manual_seed(1)
        X, y = make_classification(
            n_samples=100,
            n_features=1,
            n_redundant=0,
            n_informative=1,
            random_state=1,
            n_clusters_per_class=1,
        )
        self.X, self.y = torch.Tensor(X), torch.Tensor(y)

    def test_1d_classification_gpu(self):
        """
        Just see if we memorize the training set but on gpu
        """
        X, y = self.X, self.y
        inducing_size = 10

        model = GPClassificationModel(
            dim=1,
            inducing_size=inducing_size,
        ).to("cuda")

        model.fit(X[:50], y[:50])

        # pspace
        pm, _ = model.predict_probability(X[:50])
        pred = (pm > 0.5).cpu().numpy()
        npt.assert_allclose(pred, y[:50].cpu().numpy())

        # fspace
        pm, _ = model.predict(X[:50], probability_space=False)
        pred = (pm > 0).cpu().numpy()
        npt.assert_allclose(pred, y[:50].cpu().numpy())

        # smoke test update
        model.update(X, y)

        # pspace
        pm, _ = model.predict_probability(X)
        pred = (pm > 0.5).cpu().numpy()
        npt.assert_allclose(pred, y.cpu().numpy())

        # fspace
        pm, _ = model.predict(X, probability_space=False)
        pred = (pm > 0).cpu().numpy()
        npt.assert_allclose(pred, y.cpu().numpy())


if __name__ == "__main__":
    unittest.main()
