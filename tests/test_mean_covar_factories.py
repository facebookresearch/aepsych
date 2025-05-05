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
    DefaultMeanCovarFactory,
    PairwiseMeanCovarFactory,
    SongMeanCovarFactory,
)
from aepsych.kernels.pairwisekernel import PairwiseKernel


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
        config_str = """
        [common]
        parnames = [x]
        stimuli_per_trial = 1
        outcome_type = binary

        [x]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1
        """

        config = Config(config_str=config_str)
        factory = DefaultMeanCovarFactory.from_config(config)
        meanfun = factory.get_mean()
        covarfun = factory.get_covar()
        self._test_mean_covar(meanfun, covarfun)

    def test_default_factory_1d_dim(self):
        factory = DefaultMeanCovarFactory(dim=1)
        meanfun = factory.get_mean()
        covarfun = factory.get_covar()
        self._test_mean_covar(meanfun, covarfun)

    def test_default_factory_args_1d(self):
        config_str = """
            [common]
            parnames = [x]
            stimuli_per_trial = 1
            outcome_type = binary

            [x]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [DefaultMeanCovarFactory]
            fixed_mean = True
            lengthscale_prior = gamma
            fixed_kernel_amplitude = False
            outputscale_prior = gamma
            target = 0.5
            cov_kernel = MaternKernel
        """

        config = Config(config_str=config_str)
        factory = DefaultMeanCovarFactory.from_config(config)
        meanfun = factory.get_mean()
        covarfun = factory.get_covar()
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

    def test_default_factory_config_raises(self):
        bad_confs = [
            """
            [common]
            parnames = [x]
            stimuli_per_trial = 1
            outcome_type = binary

            [x]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [DefaultMeanCovarFactory]
            lengthscale_prior = box
            """,
            """
            [common]
            parnames = [x]
            stimuli_per_trial = 1
            outcome_type = binary

            [x]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [DefaultMeanCovarFactory]
            fixed_kernel_amplitude = False
            outputscale_prior = normal
            """,
        ]

        for conf in bad_confs:
            config = Config(config_str=conf)
            with self.assertRaises(RuntimeError):
                _ = DefaultMeanCovarFactory.from_config(config)

    def test_default_factory_init_raises(self):
        with self.assertRaises(RuntimeError):
            DefaultMeanCovarFactory(
                dim=1,
                lengthscale_prior="box",
            )

        with self.assertRaises(RuntimeError):
            DefaultMeanCovarFactory(
                dim=1,
                fixed_kernel_amplitude=False,
                outputscale_prior="normal",
            )

    def test_default_factory_2d(self):
        config_str = """
        [common]
        parnames = [x, y]
        stimuli_per_trial = 2
        outcome_type = binary

        [x]
        par_type = continuous
        lower_bound = -2
        upper_bound = 1

        [y]
        par_type = continuous
        lower_bound = 3
        upper_bound = 10
        """

        config = Config(config_str=config_str)
        factory = DefaultMeanCovarFactory.from_config(config)
        meanfun = factory.get_mean()
        covarfun = factory.get_covar()
        self.assertTrue(covarfun.base_kernel.ard_num_dims == 2)
        self.assertTrue(isinstance(meanfun, gpytorch.means.ConstantMean))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.ScaleKernel))
        self.assertTrue(isinstance(covarfun.base_kernel, gpytorch.kernels.RBFKernel))

    def test_song_factory_1d(self):
        config_str = """
        [common]
        parnames = [x]
        stimuli_per_trial = 1
        outcome_type = binary

        [x]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1
        """

        config = Config(config_str=config_str)
        factory = SongMeanCovarFactory.from_config(config)
        meanfun = factory.get_mean()
        covarfun = factory.get_covar()

        self.assertTrue(isinstance(meanfun, gpytorch.means.ConstantMean))
        self.assertTrue(isinstance(covarfun, gpytorch.kernels.AdditiveKernel))
        self.assertTrue(isinstance(covarfun.kernels[0], gpytorch.kernels.ScaleKernel))
        self.assertTrue(
            isinstance(covarfun.kernels[0].base_kernel, gpytorch.kernels.LinearKernel)
        )

    def test_song_factory_1d_intensity_RBF(self):
        config_str = """
        [common]
        parnames = [x]
        stimuli_per_trial = 1
        outcome_type = binary

        [x]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [SongMeanCovarFactory]
        intensity_RBF = True
        """

        config = Config(config_str=config_str)
        factory = SongMeanCovarFactory.from_config(config)
        meanfun = factory.get_mean()
        covarfun = factory.get_covar()

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
        config_str = """
        [common]
        parnames = [x, y]
        stimuli_per_trial = 1
        outcome_type = binary

        [x]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [y]
        par_type = continuous
        lower_bound = 1
        upper_bound = 70

        [SongMeanCovarFactory]
        target = 0.75
        """

        config = Config(config_str=config_str)
        factory = SongMeanCovarFactory.from_config(config)
        meanfun = factory.get_mean()
        covarfun = factory.get_covar()

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
        config_str = """
        [common]
        parnames = [x, y]
        stimuli_per_trial = 1
        outcome_type = binary

        [x]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [y]
        par_type = continuous
        lower_bound = 1
        upper_bound = 70

        [SongMeanCovarFactory]
        target = 0.75
        stim_dim = 0
        """

        config = Config(config_str=config_str)
        factory = SongMeanCovarFactory.from_config(config)
        meanfun = factory.get_mean()
        covarfun = factory.get_covar()

        self.assertTrue(covarfun.kernels[1].base_kernel.active_dims == 0)
        self.assertTrue(covarfun.kernels[0].base_kernel.active_dims == 1)

    def test_song_factory_2d_intensity_RBF(self):
        config_str = """
        [common]
        parnames = [x, y]
        stimuli_per_trial = 1
        outcome_type = binary

        [x]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [y]
        par_type = continuous
        lower_bound = 1
        upper_bound = 70

        [SongMeanCovarFactory]
        target = 0.75
        intensity_RBF = True
        """

        config = Config(config_str=config_str)
        factory = SongMeanCovarFactory.from_config(config)
        meanfun = factory.get_mean()
        covarfun = factory.get_covar()

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
        config_str = """
        [common]
        parnames = [x, y]
        stimuli_per_trial = 1
        outcome_type = binary

        [x]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [y]
        par_type = continuous
        lower_bound = 1
        upper_bound = 70

        [SongMeanCovarFactory]
        target = 0.75
        intensity_RBF = True
        stim_dim = 0
        """

        config = Config(config_str=config_str)
        factory = SongMeanCovarFactory.from_config(config)
        meanfun = factory.get_mean()
        covarfun = factory.get_covar()

        self.assertTrue(covarfun.kernels[1].base_kernel.active_dims == 0)
        self.assertTrue(
            np.allclose(covarfun.kernels[0].base_kernel.active_dims, [0, 1])
        )

    def test_pairwise_factory_1d(self):
        config_str = """
        [common]
        parnames = [x]
        stimuli_per_trial = 1
        outcome_type = binary

        [x]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1
        """
        config = Config(config_str=config_str)
        with self.assertRaises(ValueError):
            _ = PairwiseMeanCovarFactory.from_config(config)

    def test_pairwise_factory_2d(self):
        config_str = """
        [common]
        parnames = [x1, x2]
        stimuli_per_trial = 1
        outcome_type = binary

        [x1]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [x2]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1
        """
        config = Config(config_str=config_str)
        factory = PairwiseMeanCovarFactory.from_config(config)
        meanfun = factory.get_mean()
        covarfun = factory.get_covar()

        self.assertTrue(covarfun.latent_kernel.ard_num_dims == 1)
        self.assertIsInstance(meanfun, gpytorch.means.ZeroMean)
        self.assertIsInstance(covarfun, PairwiseKernel)
        self.assertIsInstance(covarfun.latent_kernel, gpytorch.kernels.RBFKernel)

    def test_pairwise_factory_3d(self):
        config_str = """
        [common]
        parnames = [x1, x2, x3]
        stimuli_per_trial = 1
        outcome_type = binary

        [x1]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [x2]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [x3]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1
        """
        config = Config(config_str=config_str)
        with self.assertRaises(ValueError):
            _ = PairwiseMeanCovarFactory.from_config(config)

    def test_pairwise_factory_shared(self):
        config_str = """
        [common]
        parnames = [x1, x2, x3]
        stimuli_per_trial = 1
        outcome_type = binary

        [x1]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [x2]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [x3]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [PairwiseMeanCovarFactory]
        shared_dims = 0
        zero_mean = False
        """
        config = Config(config_str=config_str)
        factory = PairwiseMeanCovarFactory.from_config(config)
        meanfun = factory.get_mean()
        covarfun = factory.get_covar()

        self.assertIsInstance(meanfun, gpytorch.means.ConstantMean)
        self.assertIsInstance(covarfun, gpytorch.kernels.ProductKernel)
        self.assertEqual(len(covarfun.kernels), 2)

        pairwise, rbf = covarfun.kernels
        self.assertIsInstance(rbf, gpytorch.kernels.RBFKernel)
        self.assertIsInstance(pairwise, PairwiseKernel)
        self.assertIsInstance(pairwise.latent_kernel, gpytorch.kernels.RBFKernel)
        self.assertTrue(pairwise.latent_kernel.ard_num_dims == 1)
