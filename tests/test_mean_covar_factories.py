#!/usr/bin/env python3
import unittest

import gpytorch
from aepsych.config import Config
from aepsych.factory import (
    default_mean_covar_factory,
    monotonic_mean_covar_factory,
    song_mean_covar_factory,
)
from aepsych.kernels.rbf_partial_grad import (
    RBFKernelPartialObsGrad,
)
from aepsych.means.constant_partial_grad import (
    ConstantMeanPartialObsGrad,
)
from scipy.stats import norm


class TestFactories(unittest.TestCase):
    def test_default_factory_1d(self):

        conf = {"default_mean_covar_factory": {"lb": [0], "ub": [1]}}
        config = Config(config_dict=conf)
        meanfun, covarfun = default_mean_covar_factory(config)
        self.assertTrue(covarfun.base_kernel.ard_num_dims == 1)
        self.assertTrue(isinstance(meanfun, gpytorch.means.ConstantMean))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.ScaleKernel))
        self.assertTrue(isinstance(covarfun.base_kernel, gpytorch.kernels.RBFKernel))

    def test_default_factory_2d(self):
        conf = {"default_mean_covar_factory": {"lb": [-2, 3], "ub": [1, 10]}}
        config = Config(config_dict=conf)
        meanfun, covarfun = default_mean_covar_factory(config)
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
        self.assertTrue(meanfun.constant.requires_grad is False)
        self.assertTrue(meanfun.constant == norm.ppf(0.75))

    def test_monotonic_factory_2d(self):
        conf = {
            "monotonic_mean_covar_factory": {
                "lb": [0, 1],
                "ub": [1, 70],
                "target": 0.89,
            }
        }
        config = Config(config_dict=conf)
        meanfun, covarfun = monotonic_mean_covar_factory(config)
        self.assertTrue(covarfun.base_kernel.ard_num_dims == 2)
        self.assertTrue(isinstance(meanfun, ConstantMeanPartialObsGrad))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.ScaleKernel))
        self.assertTrue(isinstance(covarfun.base_kernel, RBFKernelPartialObsGrad))
        self.assertTrue(meanfun.constant.requires_grad is False)
        self.assertTrue(meanfun.constant == norm.ppf(0.89))

    def test_song_factory_1d(self):
        conf = {"song_mean_covar_factory": {"lb": [0], "ub": [1]}}
        config = Config(config_dict=conf)
        meanfun, covarfun = song_mean_covar_factory(config)
        self.assertTrue(covarfun.kernels[0].base_kernel.ard_num_dims == 1)
        self.assertTrue(isinstance(meanfun, gpytorch.means.ConstantMean))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.AdditiveKernel))
        self.assertTrue(isinstance(covarfun.kernels[0], gpytorch.kernels.ScaleKernel))
        self.assertTrue(
            isinstance(covarfun.kernels[0].base_kernel, gpytorch.kernels.LinearKernel)
        )

    def test_song_factory_2d(self):
        conf = {
            "song_mean_covar_factory": {"lb": [0, 1], "ub": [1, 70], "target": 0.75}
        }
        config = Config(config_dict=conf)
        meanfun, covarfun = song_mean_covar_factory(config)
        self.assertTrue(covarfun.kernels[0].base_kernel.ard_num_dims == 1)
        self.assertTrue(covarfun.kernels[1].base_kernel.ard_num_dims == 1)
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
