#!/usr/bin/env python3
import unittest

import gpytorch
from aepsych.config import Config
from aepsych.kernels import PairwiseKernel
from aepsych.factory import (
    pairwise_kernel_factory,
)
from aepsych.means.constant_partial_grad import (
    ConstantMeanPartialObsGrad,
)
from aepsych.kernels.rbf_partial_grad import (
    RBFKernelPartialObsGrad,
)


class TestFactories(unittest.TestCase):
    def test_pairwise_factory_1d(self):
        conf = {
            "common": {
                "lb": [0],
                "ub": [1],
                "stimuli_per_trial":  2,
            }}
        config = Config(config_dict=conf)
        latent_mean, pw_covar = pairwise_kernel_factory(config=config)

        self.assertTrue(isinstance(latent_mean, gpytorch.means.ConstantMean))
        self.assertTrue(isinstance(pw_covar, PairwiseKernel))
        self.assertTrue(isinstance(pw_covar.latent_kernel, gpytorch.kernels.ScaleKernel))
        self.assertTrue(isinstance(pw_covar.latent_kernel.base_kernel, gpytorch.kernels.RBFKernel))
        self.assertTrue(pw_covar.latent_kernel.base_kernel.ard_num_dims == 1)
        self.assertTrue(
            isinstance(
                pw_covar.latent_kernel.base_kernel._priors["lengthscale_prior"][0],
                gpytorch.priors.GammaPrior,
            )
        )
        self.assertTrue(
            isinstance(
                pw_covar.latent_kernel._priors["outputscale_prior"][0],
                gpytorch.priors.SmoothedBoxPrior,
            )
        )

    def test_pairwise_factory_3d(self):
        conf = {
            "common": {
                "lb": [0, -1, -2],
                "ub": [1, 1, -1],
                "stimuli_per_trial":  2,
            }}
        config = Config(config_dict=conf)
        latent_mean, pw_covar = pairwise_kernel_factory(config=config)

        self.assertTrue(isinstance(latent_mean, gpytorch.means.ConstantMean))
        self.assertTrue(isinstance(pw_covar, PairwiseKernel))
        self.assertTrue(isinstance(pw_covar.latent_kernel, gpytorch.kernels.ScaleKernel))
        self.assertTrue(isinstance(pw_covar.latent_kernel.base_kernel, gpytorch.kernels.RBFKernel))
        self.assertTrue(pw_covar.latent_kernel.base_kernel.ard_num_dims == 3)
        self.assertTrue(
            isinstance(
                pw_covar.latent_kernel.base_kernel._priors["lengthscale_prior"][0],
                gpytorch.priors.GammaPrior,
            )
        )
        self.assertTrue(
            isinstance(
                pw_covar.latent_kernel._priors["outputscale_prior"][0],
                gpytorch.priors.SmoothedBoxPrior,
            )
        )

    def test_pairwise_factory_latent(self):
        conf = {
            "common": {
                "lb": [0],
                "ub": [1],
                "stimuli_per_trial":  2,
            },
            "pairwise_kernel_factory": {
                "latent_factory": "monotonic_mean_covar_factory",
            },
        }
        config = Config(config_dict=conf)
        latent_mean, pw_covar = pairwise_kernel_factory(config=config)

        self.assertTrue(isinstance(latent_mean, ConstantMeanPartialObsGrad))
        self.assertTrue(latent_mean.constant.requires_grad)
        self.assertTrue(pw_covar.is_partial_obs)
        self.assertTrue(isinstance(pw_covar, PairwiseKernel))
        self.assertTrue(isinstance(pw_covar.latent_kernel, gpytorch.kernels.ScaleKernel))
        self.assertTrue(isinstance(pw_covar.latent_kernel.base_kernel, RBFKernelPartialObsGrad))
        self.assertTrue(pw_covar.latent_kernel.base_kernel.ard_num_dims == 1)
        self.assertTrue(
            isinstance(
                pw_covar.latent_kernel.base_kernel._priors["lengthscale_prior"][0],
                gpytorch.priors.GammaPrior,
            )
        )
        self.assertTrue(
            isinstance(
                pw_covar.latent_kernel._priors["outputscale_prior"][0],
                gpytorch.priors.SmoothedBoxPrior,
            )
        )

if __name__ == "__main__":
    unittest.main()
