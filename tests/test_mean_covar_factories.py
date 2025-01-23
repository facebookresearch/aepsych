#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import gpytorch
import numpy as np
from aepsych.config import Config
from aepsych.factory import (
    default_mean_covar_factory,
    monotonic_mean_covar_factory,
    pairwise_mean_covar_factory,
    song_mean_covar_factory,
)
from aepsych.kernels.pairwisekernel import PairwiseKernel
from aepsych.kernels.rbf_partial_grad import RBFKernelPartialObsGrad
from aepsych.means.constant_partial_grad import ConstantMeanPartialObsGrad
from scipy.stats import norm


class TestFactories(unittest.TestCase):
    def _test_mean_covar(self, meanfun, covarfun):
        self.assertTrue(covarfun.ard_num_dims == 1)
        self.assertTrue(meanfun.constant.requires_grad)
        self.assertTrue(isinstance(meanfun, gpytorch.means.ConstantMean))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.RBFKernel))
        self.assertTrue(
            isinstance(
                covarfun._priors["lengthscale_prior"][0],
                gpytorch.priors.LogNormalPrior,
            )
        )

    def test_default_factory_1d_config(self):
        config = Config(
            config_dict={"default_mean_covar_factory": {"lb": [0], "ub": [1]}}
        )
        meanfun, covarfun = default_mean_covar_factory(config=config)
        self._test_mean_covar(meanfun, covarfun)

    def test_default_factory_1d_dim(self):
        meanfun, covarfun = default_mean_covar_factory(dim=1)
        self._test_mean_covar(meanfun, covarfun)

    def test_default_factory_args_1d(self):
        conf = {
            "default_mean_covar_factory": {
                "lb": [0],
                "ub": [1],
                "fixed_mean": True,
                "lengthscale_prior": "gamma",
                "fixed_kernel_amplitude": False,
                "outputscale_prior": "gamma",
                "target": 0.5,
                "kernel": "MaternKernel",
            }
        }
        config = Config(config_dict=conf)
        meanfun, covarfun = default_mean_covar_factory(config)
        self.assertFalse(meanfun.constant.requires_grad)
        self.assertTrue(isinstance(meanfun, gpytorch.means.ConstantMean))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.ScaleKernel))
        self.assertTrue(
            isinstance(
                covarfun.base_kernel._priors["lengthscale_prior"][0],
                gpytorch.priors.GammaPrior,
            )
        )
        self.assertTrue(
            isinstance(
                covarfun._priors["outputscale_prior"][0], gpytorch.priors.GammaPrior
            )
        )

        self.assertTrue(
            covarfun.base_kernel._priors["lengthscale_prior"][0].concentration == 3.0
        )
        self.assertTrue(
            covarfun.base_kernel._priors["lengthscale_prior"][0].rate == 6.0
        )
        self.assertTrue(covarfun._priors["outputscale_prior"][0].concentration == 2.0)
        self.assertTrue(covarfun._priors["outputscale_prior"][0].rate == 0.15)
        self.assertTrue(
            covarfun.base_kernel._priors["lengthscale_prior"][0]._transform is None
        )
        self.assertTrue(isinstance(covarfun.base_kernel, gpytorch.kernels.MaternKernel))

    def test_default_factory_raises(self):
        bad_confs = [
            {
                "default_mean_covar_factory": {
                    "lb": [0],
                    "ub": [1],
                    "lengthscale_prior": "box",
                }
            },
            {
                "default_mean_covar_factory": {
                    "lb": [0],
                    "ub": [1],
                    "fixed_kernel_amplitude": False,
                    "outputscale_prior": "normal",
                }
            },
            {"default_mean_covar_factory": {"lb": [0], "ub": [1], "fixed_mean": True}},
        ]
        for conf in bad_confs:
            with self.assertRaises(RuntimeError):
                config = Config(conf)
                _, __ = default_mean_covar_factory(config)

        with self.assertRaises(AssertionError):
            default_mean_covar_factory()

        config = Config(
            config_dict={"default_mean_covar_factory": {"lb": [0], "ub": [1]}}
        )
        with self.assertRaises(AssertionError):
            default_mean_covar_factory(config=config, dim=2)

    def test_default_factory_2d(self):
        conf = {"default_mean_covar_factory": {"lb": [-2, 3], "ub": [1, 10]}}
        config = Config(config_dict=conf)
        meanfun, covarfun = default_mean_covar_factory(config, stimuli_per_trial=2)
        self.assertTrue(covarfun.base_kernel.ard_num_dims == 2)
        self.assertTrue(isinstance(meanfun, gpytorch.means.ConstantMean))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.ScaleKernel))
        self.assertTrue(isinstance(covarfun.base_kernel, gpytorch.kernels.RBFKernel))

    def test_monotonic_factory_1d(self):
        conf = {"monotonic_mean_covar_factory": {"lb": [0], "ub": [1]}}
        config = Config(config_dict=conf)
        meanfun, covarfun = monotonic_mean_covar_factory(config)
        self.assertTrue(covarfun.base_kernel.ard_num_dims == 1)
        self.assertTrue(isinstance(meanfun, ConstantMeanPartialObsGrad))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.ScaleKernel))
        self.assertTrue(isinstance(covarfun.base_kernel, RBFKernelPartialObsGrad))
        self.assertTrue(meanfun.constant.requires_grad)

    def test_monotonic_factory_args_1d(self):
        conf = {
            "monotonic_mean_covar_factory": {
                "lb": [0],
                "ub": [1],
                "fixed_mean": True,
                "target": 0.88,
            }
        }
        config = Config(config_dict=conf)

        meanfun, covarfun = monotonic_mean_covar_factory(config)
        self.assertTrue(covarfun.base_kernel.ard_num_dims == 1)
        self.assertTrue(isinstance(meanfun, ConstantMeanPartialObsGrad))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.ScaleKernel))
        self.assertTrue(isinstance(covarfun.base_kernel, RBFKernelPartialObsGrad))
        self.assertFalse(meanfun.constant.requires_grad)
        self.assertTrue(np.allclose(meanfun.constant, norm.ppf(0.88)))

    def test_monotonic_factory_2d(self):
        conf = {
            "monotonic_mean_covar_factory": {
                "lb": [0, 1],
                "ub": [1, 70],
                "fixed_mean": True,
                "target": 0.89,
            }
        }
        config = Config(config_dict=conf)
        meanfun, covarfun = monotonic_mean_covar_factory(config)
        self.assertTrue(covarfun.base_kernel.ard_num_dims == 2)
        self.assertTrue(isinstance(meanfun, ConstantMeanPartialObsGrad))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.ScaleKernel))
        self.assertTrue(isinstance(covarfun.base_kernel, RBFKernelPartialObsGrad))
        self.assertFalse(meanfun.constant.requires_grad)
        self.assertTrue(np.allclose(meanfun.constant, norm.ppf(0.89)))

    def test_song_factory_1d(self):
        conf = {"song_mean_covar_factory": {"lb": [0], "ub": [1]}}
        config = Config(config_dict=conf)
        meanfun, covarfun = song_mean_covar_factory(config)
        self.assertTrue(isinstance(meanfun, gpytorch.means.ConstantMean))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.AdditiveKernel))
        self.assertTrue(isinstance(covarfun.kernels[0], gpytorch.kernels.ScaleKernel))
        self.assertTrue(
            isinstance(covarfun.kernels[0].base_kernel, gpytorch.kernels.LinearKernel)
        )

    def test_song_factory_1d_intensity_RBF(self):
        conf = {
            "song_mean_covar_factory": {"lb": [0], "ub": [1], "intensity_RBF": True}
        }
        config = Config(config_dict=conf)
        meanfun, covarfun = song_mean_covar_factory(config)
        self.assertTrue(covarfun.kernels[0].base_kernel.ard_num_dims == 1)
        self.assertTrue(isinstance(meanfun, gpytorch.means.ConstantMean))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.AdditiveKernel))
        self.assertTrue(isinstance(covarfun.kernels[0], gpytorch.kernels.ScaleKernel))
        self.assertTrue(isinstance(covarfun.kernels[1], gpytorch.kernels.ScaleKernel))
        self.assertTrue(
            isinstance(covarfun.kernels[0].base_kernel, gpytorch.kernels.RBFKernel)
        )
        self.assertTrue(
            isinstance(covarfun.kernels[1].base_kernel, gpytorch.kernels.LinearKernel)
        )

    def test_song_factory_2d(self):
        conf = {
            "song_mean_covar_factory": {"lb": [0, 1], "ub": [1, 70], "target": 0.75}
        }
        config = Config(config_dict=conf)
        meanfun, covarfun = song_mean_covar_factory(config)
        self.assertEqual(covarfun.kernels[0].base_kernel.ard_num_dims, 1)
        self.assertTrue(isinstance(meanfun, gpytorch.means.ConstantMean))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.AdditiveKernel))
        self.assertTrue(isinstance(covarfun.kernels[0], gpytorch.kernels.ScaleKernel))
        self.assertTrue(isinstance(covarfun.kernels[1], gpytorch.kernels.ScaleKernel))
        self.assertTrue(
            isinstance(covarfun.kernels[0].base_kernel, gpytorch.kernels.RBFKernel)
        )
        self.assertTrue(covarfun.kernels[0].base_kernel.active_dims == 0)
        self.assertTrue(
            isinstance(covarfun.kernels[1].base_kernel, gpytorch.kernels.LinearKernel)
        )
        self.assertTrue(covarfun.kernels[1].base_kernel.active_dims == 1)

        # flip the stim dim
        conf = {
            "song_mean_covar_factory": {
                "lb": [0, 1],
                "ub": [1, 70],
                "target": 0.75,
                "stim_dim": 0,
            }
        }
        config = Config(config_dict=conf)
        meanfun, covarfun = song_mean_covar_factory(config)
        self.assertTrue(covarfun.kernels[1].base_kernel.active_dims == 0)
        self.assertTrue(covarfun.kernels[0].base_kernel.active_dims == 1)

    def test_song_factory_2d_intensity_RBF(self):
        conf = {
            "song_mean_covar_factory": {
                "lb": [0, 1],
                "ub": [1, 70],
                "target": 0.75,
                "intensity_RBF": True,
            }
        }
        config = Config(config_dict=conf)
        meanfun, covarfun = song_mean_covar_factory(config)
        self.assertEqual(covarfun.kernels[0].base_kernel.ard_num_dims, 2)
        self.assertTrue(isinstance(meanfun, gpytorch.means.ConstantMean))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.AdditiveKernel))
        self.assertTrue(isinstance(covarfun.kernels[0], gpytorch.kernels.ScaleKernel))
        self.assertTrue(isinstance(covarfun.kernels[1], gpytorch.kernels.ScaleKernel))
        self.assertTrue(
            isinstance(covarfun.kernels[0].base_kernel, gpytorch.kernels.RBFKernel)
        )
        self.assertTrue(
            np.allclose(covarfun.kernels[0].base_kernel.active_dims, [0, 1])
        )
        self.assertTrue(
            isinstance(covarfun.kernels[1].base_kernel, gpytorch.kernels.LinearKernel)
        )
        self.assertEqual(covarfun.kernels[1].base_kernel.active_dims, 1)

        # flip the stim dim
        conf = {
            "song_mean_covar_factory": {
                "lb": [0, 1],
                "ub": [1, 70],
                "target": 0.75,
                "stim_dim": 0,
                "intensity_RBF": True,
            }
        }
        config = Config(config_dict=conf)
        meanfun, covarfun = song_mean_covar_factory(config)
        self.assertTrue(covarfun.kernels[1].base_kernel.active_dims == 0)
        self.assertTrue(
            np.allclose(covarfun.kernels[0].base_kernel.active_dims, [0, 1])
        )

    def test_pairwise_factory_1d(self):
        conf = {"common": {"lb": [0], "ub": [1]}}
        config = Config(config_dict=conf)
        with self.assertRaises(AssertionError):
            meanfun, covarfun = pairwise_mean_covar_factory(config)

    def test_pairwise_factory_2d(self):
        conf = {"common": {"lb": [0, 0], "ub": [1, 1]}}
        config = Config(config_dict=conf)
        meanfun, covarfun = pairwise_mean_covar_factory(config)
        self.assertTrue(covarfun.latent_kernel.ard_num_dims == 1)
        self.assertIsInstance(meanfun, gpytorch.means.ZeroMean)
        self.assertIsInstance(covarfun, PairwiseKernel)
        self.assertIsInstance(covarfun.latent_kernel, gpytorch.kernels.RBFKernel)

    def test_pairwise_factory_3d(self):
        conf = {"common": {"lb": [0, 0, 0], "ub": [1, 1, 1]}}
        config = Config(config_dict=conf)
        with self.assertRaises(AssertionError):
            meanfun, covarfun = pairwise_mean_covar_factory(config)

    def test_pairwise_factory_shared(self):
        conf = {
            "common": {"lb": [0, 0, 0], "ub": [1, 1, 1]},
            "pairwise_mean_covar_factory": {"shared_dims": [0], "zero_mean": False},
        }
        config = Config(config_dict=conf)
        meanfun, covarfun = pairwise_mean_covar_factory(config)
        self.assertIsInstance(meanfun, gpytorch.means.ConstantMean)
        self.assertIsInstance(covarfun, gpytorch.kernels.ProductKernel)
        self.assertEqual(len(covarfun.kernels), 2)

        pairwise, rbf = covarfun.kernels
        self.assertIsInstance(rbf, gpytorch.kernels.RBFKernel)
        self.assertIsInstance(pairwise, PairwiseKernel)
        self.assertIsInstance(pairwise.latent_kernel, gpytorch.kernels.RBFKernel)
        self.assertTrue(pairwise.latent_kernel.ard_num_dims == 1)
