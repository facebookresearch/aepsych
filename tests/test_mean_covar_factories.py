#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import gpytorch
import numpy as np
import torch
from aepsych.config import Config
from aepsych.factory import (
    DefaultMeanCovarFactory,
    MixedMeanCovarFactory,
    PairwiseMeanCovarFactory,
    SongMeanCovarFactory,
)
from aepsych.generators import SobolGenerator
from aepsych.kernels.pairwisekernel import PairwiseKernel
from aepsych.models import GPClassificationModel
from aepsych.strategy import SequentialStrategy
from scipy.stats import bernoulli, norm


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


class TestMixedFactories(unittest.TestCase):
    @staticmethod
    def new_novel_det_channels_params(
        channel, scale_factor=1.0, wave_freq=1, target=0.75
    ):
        """Get the target parameters for 2D synthetic novel_det(channel) function
            Keyword arguments:
        channel -- 1D array of channel locations whose thresholds to return
        scale factor -- scale for the novel_det function, where higher is steeper/lower SD
        wave_freq -- frequency of location waveform on [-1,1]
        target -- target threshold
        """
        locs = -0.3 * np.sin(5 * wave_freq * (channel - 1 / 6) / np.pi) ** 2 - 0.5
        scale = (
            1
            / (10 * scale_factor)
            * (0.75 + 0.25 * np.cos(10 * (0.3 + channel) / np.pi))
        )
        return locs, scale

    @staticmethod
    def target_new_novel_det_channels(
        channel, scale_factor=1.0, wave_freq=1, target=0.75
    ):
        """Get the target (i.e. threshold) for 2D synthetic novel_det(channel) function
            Keyword arguments:
        channel -- 1D array of channel locations whose thresholds to return
        scale factor -- scale for the novel_det function, where higher is steeper/lower SD
        wave_freq -- frequency of location waveform on [-1,1]
        target -- target threshold
        """
        locs, scale = TestMixedFactories.new_novel_det_channels_params(
            channel, scale_factor, wave_freq, target
        )
        return norm.ppf(target, loc=locs, scale=scale)

    @staticmethod
    def new_novel_det_channels(x, channel, scale_factor=1.0, wave_freq=1, target=0.75):
        """Get the 2D synthetic novel_det(channel) function
            Keyword arguments:
        x -- array of shape (n,2) of locations to sample;
            x[...,0] is channel from -1 to 1; x[...,1] is intensity from -1 to 1
        scale factor -- scale for the novel_det function, where higher is steeper/lower SD
        wave_freq -- frequency of location waveform on [-1,1]
        """
        locs, scale = TestMixedFactories.new_novel_det_channels_params(
            channel, scale_factor, wave_freq, target
        )
        return (x - locs) / scale

    @staticmethod
    def cdf_new_novel_det_channels(channel, scale_factor=1.0, wave_freq=1, target=0.75):
        """Get the cdf for 2D synthetic novel_det(channel) function
            Keyword arguments:
        x -- array of shape (n,2) of locations to sample;
            x[...,0] is channel from -1 to 1; x[...,1] is intensity from -1 to 1
        scale factor -- scale for the novel_det function, where higher is steeper/lower SD
        wave_freq -- frequency of location waveform on [-1,1]
        """
        return norm.cdf(
            TestMixedFactories.new_novel_det_channels(
                channel, scale_factor, wave_freq, target
            )
        )

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        n_train = 20
        n_test = 10
        generator = SobolGenerator(lb=[-1], ub=[1], dim=1)
        x = generator.gen(n_train)
        channel = np.random.choice(2, n_train)
        f = TestMixedFactories.new_novel_det_channels(x.squeeze(), channel)
        y = bernoulli.rvs(norm.cdf(f))
        self.xtest = generator.gen(n_test)
        self.x = torch.concatenate([x, torch.tensor(channel).unsqueeze(1)], axis=1)
        self.y = torch.Tensor(y)

        self.f1_true = TestMixedFactories.new_novel_det_channels(
            self.xtest.squeeze(), 0
        )
        self.f2_true = TestMixedFactories.new_novel_det_channels(
            self.xtest.squeeze(), 1
        )

    def test_categorical_classification_smoke(self):
        factory = MixedMeanCovarFactory(
            dim=2, discrete_params={1: 2}, discrete_kernel="categorical"
        )
        mean = factory.get_mean()
        covar = factory.get_covar()
        model = GPClassificationModel(dim=2, mean_module=mean, covar_module=covar)

        model.fit(self.x, self.y)

        p1_pred, _ = model.predict_probability(
            torch.concatenate([self.xtest, torch.zeros_like(self.xtest)], axis=1),
        )
        num_mismatches = (
            (p1_pred > 0.5).numpy() != (norm.cdf(self.f1_true.squeeze()) > 0.5)
        ).sum()
        self.assertLessEqual(num_mismatches, 1)

        p2_pred, _ = model.predict_probability(
            torch.concatenate([self.xtest, torch.ones_like(self.xtest)], axis=1),
        )
        num_mismatches = (
            (p2_pred > 0.5).numpy() != (norm.cdf(self.f2_true.squeeze()) > 0.5)
        ).sum()
        self.assertLessEqual(num_mismatches, 1)

    def test_index_classification_smoke(self):
        factory = MixedMeanCovarFactory(
            dim=2, discrete_params={1: 2}, discrete_kernel="index"
        )
        mean = factory.get_mean()
        covar = factory.get_covar()
        model = GPClassificationModel(dim=2, mean_module=mean, covar_module=covar)

        model.fit(self.x, self.y)

        p1_pred, _ = model.predict_probability(
            torch.concatenate([self.xtest, torch.zeros_like(self.xtest)], axis=1),
        )
        num_mismatches = (
            (p1_pred > 0.5).numpy() != (norm.cdf(self.f1_true.squeeze()) > 0.5)
        ).sum()
        self.assertLessEqual(num_mismatches, 1)

        p2_pred, _ = model.predict_probability(
            torch.concatenate([self.xtest, torch.ones_like(self.xtest)], axis=1),
        )
        num_mismatches = (
            (p2_pred > 0.5).numpy() != (norm.cdf(self.f2_true.squeeze()) > 0.5)
        ).sum()
        self.assertLessEqual(num_mismatches, 1)

    def test_mixed_from_config(self):
        config_str = """
        [common]
        parnames = [x, channel, color, y]
        stimuli_per_trial = 1
        outcome_types = [binary]
        strategy_names = [init_strat, opt_strat]

        [x]
        par_type = continuous
        lower_bound = -1
        upper_bound = 1

        [y]
        par_type = continuous
        lower_bound = -1
        upper_bound = 1

        [channel]
        par_type = categorical
        choices = [left, middle, right]
        rank = 2

        [color]
        par_type = categorical
        choices = [red, green, blue]
        rank = 3

        [init_strat]
        generator = SobolGenerator
        min_asks = 20

        [opt_strat]
        generator = OptimizeAcqfGenerator
        model = GPClassificationModel
        min_asks = 1

        [OptimizeAcqfGenerator]
        acqf = qLogNoisyExpectedImprovement

        [GPClassificationModel]
        mean_covar_factory = MixedMeanCovarFactory

        [MixedMeanCovarFactory]
        discrete_kernel = index
        """
        config = Config(config_str=config_str)
        strat = SequentialStrategy.from_config(config)

        model = strat.strat_list[-1].model
        covar = model.covar_module

        # Basic check
        self.assertEqual(model.dim, 4)
        self.assertIsInstance(covar, gpytorch.kernels.ProductKernel)

        # Check the additive part
        add_kernel = covar.kernels[0]
        self.assertIsInstance(add_kernel.kernels[0], gpytorch.kernels.RBFKernel)
        self.assertSequenceEqual(add_kernel.kernels[0].active_dims, (0, 3))
        self.assertEqual(len(add_kernel.kernels[1:]), 2)
        for kernel, index, rank in zip(add_kernel.kernels[1:], (1, 2), (2, 3)):
            self.assertIsInstance(kernel, gpytorch.kernels.IndexKernel)
            self.assertEqual(kernel.active_dims.item(), index)
            self.assertEqual(kernel.covar_factor.shape[1], rank)

        # Check the product part
        cont_kernel = covar.kernels[1]
        self.assertIsInstance(cont_kernel, gpytorch.kernels.RBFKernel)
        self.assertSequenceEqual(cont_kernel.active_dims, (0, 3))

        index_kernels = covar.kernels[2:]
        for kernel, index, rank in zip(index_kernels, (1, 2), (2, 3)):
            self.assertIsInstance(kernel, gpytorch.kernels.IndexKernel)
            self.assertEqual(kernel.active_dims.item(), index)
            self.assertEqual(kernel.covar_factor.shape[1], rank)

        # Check there's copies and not duplicates
        self.assertNotEqual(add_kernel.kernels[0], cont_kernel)
        self.assertNotEqual(add_kernel.kernels[1], index_kernels[0])
        self.assertNotEqual(add_kernel.kernels[2], index_kernels[1])

    def test_mixed_acquisition(self):
        def f_1d(x):
            """
            latent is just a gaussian bump at mu
            """
            if len(x.shape) == 1:
                if x[1] == 0:
                    mu = 0.0
                else:
                    mu = 0.4
                return torch.exp(-((x[0] - mu) ** 2))
            else:
                results = []
                for row in x:
                    results.append(f_1d(row))

                return torch.tensor(results)

        config_str = """
        [common]
        parnames = [x, channel]
        stimuli_per_trial = 1
        outcome_types = [binary]
        strategy_names = [init_strat, opt_strat]

        [x]
        par_type = continuous
        lower_bound = -1
        upper_bound = 1

        [channel]
        par_type = categorical
        choices = [left, right]

        [init_strat]
        generator = SobolGenerator
        min_asks = 200

        [opt_strat]
        generator = MixedOptimizeAcqfGenerator
        model = GPClassificationModel
        min_asks = 1

        [MixedOptimizeAcqfGenerator]
        acqf = qLogNoisyExpectedImprovement

        [GPClassificationModel]
        mean_covar_factory = MixedMeanCovarFactory

        [MixedMeanCovarFactory]
        # discrete_kernel = index
        """
        config = Config(config_str=config_str)
        strat = SequentialStrategy.from_config(config)

        while not strat.finished:
            point = strat.gen()
            y = f_1d(point)
            response = torch.bernoulli(y)
            strat.add_data(point, response)

            print(f"{point=}, {response=}")

        x = torch.linspace(-1, 1, 11).unsqueeze(1)
        channel0 = torch.zeros_like(x)
        channel1 = torch.ones_like(x)
        x_0 = torch.cat([x, channel0], dim=1)
        x_1 = torch.cat([x, channel1], dim=1)

        y_0, _ = strat.predict(x_0, probability_space=True)
        y_1, _ = strat.predict(x_1, probability_space=True)

        self.assertTrue(torch.allclose(x[torch.argmax(y_0)], torch.tensor([0.0])))
        self.assertTrue(torch.allclose(x[torch.argmax(y_1)], torch.tensor([0.4])))
