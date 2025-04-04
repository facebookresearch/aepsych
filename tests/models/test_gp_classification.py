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

from functools import partial

import numpy as np
import numpy.testing as npt
from aepsych.acquisition import MCLevelSetEstimation
from aepsych.config import Config
from aepsych.generators import OptimizeAcqfGenerator, SobolGenerator
from aepsych.kernels import PairwiseKernel
from aepsych.models import GPClassificationModel
from aepsych.models.utils import get_jnd
from aepsych.strategy import SequentialStrategy, Strategy
from aepsych.transforms import ParameterTransformedModel, ParameterTransforms
from aepsych.transforms.ops import NormalizeScale
from botorch.acquisition import qUpperConfidenceBound
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.optim.stopping import ExpMAStoppingCriterion
from gpytorch.means import ZeroMean
from scipy.stats import bernoulli, norm, pearsonr
from sklearn.datasets import make_classification
from torch.distributions import Normal
from torch.optim import Adam


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


def f_2d_target(x, target=None):
    """
    Distance to target
    """
    if target is None:
        target = torch.tensor([0.0, 0.0])

    if x.ndim > 1:
        return torch.exp(-torch.linalg.vector_norm(x - target, dim=1))

    return torch.exp(-torch.linalg.vector_norm(x - target))


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

    def test_1d_classification(self):
        """
        Just see if we memorize the training set
        """
        X, y = self.X, self.y
        inducing_size = 10

        model = GPClassificationModel(
            dim=1,
            inducing_size=inducing_size,
        )

        model.fit(X[:50], y[:50])

        # pspace
        pm, _ = model.predict_probability(X[:50])
        pred = (pm > 0.5).numpy()
        npt.assert_allclose(pred, y[:50].numpy())

        # fspace
        pm, _ = model.predict(X[:50], probability_space=False)
        pred = (pm > 0).numpy()
        npt.assert_allclose(pred, y[:50].numpy())

        # smoke test update
        model.update(X, y)

        # pspace
        pm, _ = model.predict_probability(X)
        pred = (pm > 0.5).numpy()
        npt.assert_allclose(pred, y.numpy())

        # fspace
        pm, _ = model.predict(X, probability_space=False)
        pred = (pm > 0).numpy()
        npt.assert_allclose(pred, y.numpy())

    def test_1d_classification_pytorchopt(self):
        """
        Just see if we memorize the training set
        """
        X, y = self.X, self.y
        inducing_size = 10

        model = GPClassificationModel(
            dim=1,
            inducing_size=inducing_size,
        )

        model.fit(
            X[:50],
            y[:50],
            optimizer=fit_gpytorch_mll_torch,
            optimizer_kwargs={
                "stopping_criterion": ExpMAStoppingCriterion(maxiter=30),
                "optimizer": partial(Adam, lr=0.05),
            },
        )

        # pspace
        pm, _ = model.predict_probability(X[:50])
        pred = (pm > 0.5).numpy()
        npt.assert_allclose(pred, y[:50])

        # fspace
        pm, _ = model.predict(X[:50], probability_space=False)
        pred = (pm > 0).numpy()
        npt.assert_allclose(pred, y[:50])

        # smoke test update
        model.update(
            X,
            y,
            optimizer=fit_gpytorch_mll_torch,
            optimizer_kwargs={},
        )

        # pspace
        pm, _ = model.predict_probability(X)
        pred = (pm > 0.5).numpy()
        npt.assert_allclose(pred, y)

        # fspace
        pm, _ = model.predict(X, probability_space=False)
        pred = (pm > 0).numpy()
        npt.assert_allclose(pred, y)

    def test_1d_classification_different_scales(self):
        """
        Just see if we memorize the training set
        """
        np.random.seed(1)
        torch.manual_seed(1)
        X, y = make_classification(
            n_features=2,
            n_redundant=0,
            n_informative=1,
            random_state=1,
            n_clusters_per_class=1,
        )
        X, y = torch.Tensor(X), torch.Tensor(y)
        X[:, 0] = X[:, 0] * 1000
        X[:, 1] = X[:, 1] / 1000
        lb = torch.tensor([-3000.0, -0.003])
        ub = torch.tensor([3000.0, 0.003])
        inducing_size = 20

        transforms = ParameterTransforms(
            normalize=NormalizeScale(2, bounds=torch.stack((lb, ub)))
        )
        model = ParameterTransformedModel(
            model=GPClassificationModel,
            inducing_size=inducing_size,
            transforms=transforms,
            dim=2,
        )
        model.fit(X[:50], y[:50])

        # pspace
        pm, _ = model.predict_probability(X[:50])
        pred = (pm > 0.5).numpy()
        npt.assert_allclose(pred, y[:50])

        # fspace
        pm, _ = model.predict(X[:50], probability_space=False)
        pred = (pm > 0).numpy()
        npt.assert_allclose(pred, y[:50])

        # smoke test update
        model.update(X, y)

        # pspace
        pm, _ = model.predict_probability(X)
        pred = (pm > 0.5).numpy()
        npt.assert_allclose(pred, y)

        # fspace
        pm, _ = model.predict(X, probability_space=False)
        pred = (pm > 0).numpy()
        npt.assert_allclose(pred, y)

    def test_reset_hyperparams(self):
        model = GPClassificationModel(
            dim=1,
            inducing_size=20,
        )

        ls_before = model.covar_module.lengthscale.clone().detach().numpy()
        model.fit(torch.Tensor(self.X), torch.Tensor(self.y))

        ls_after = model.covar_module.lengthscale.clone().cpu().detach().numpy()

        model._reset_hyperparameters()

        ls_reset = model.covar_module.lengthscale.clone().cpu().detach().numpy()

        # before should be different from after and after should be different
        # from reset but before and reset should be same
        self.assertFalse(np.allclose(ls_before, ls_after))
        self.assertFalse(np.allclose(ls_after, ls_reset))
        self.assertTrue(np.allclose(ls_before, ls_reset))

    def test_reset_variational_strategy(self):
        model = GPClassificationModel(
            dim=1,
            inducing_size=20,
        )

        variational_params_before = [
            v.clone().detach().numpy() for v in model.variational_parameters()
        ]
        induc_before = model.variational_strategy.inducing_points

        model.fit(torch.Tensor(self.X), torch.Tensor(self.y))

        variational_params_after = [
            v.clone().cpu().detach().numpy() for v in model.variational_parameters()
        ]
        induc_after = model.variational_strategy.inducing_points

        model._reset_variational_strategy()

        variational_params_reset = [
            v.clone().cpu().detach().numpy() for v in model.variational_parameters()
        ]
        induc_reset = model.variational_strategy.inducing_points

        # before should be different from after and after should be different
        # from reset
        self.assertFalse(np.allclose(induc_before, induc_after))
        if (
            induc_after.shape == induc_reset.shape
        ):  # If they're not the same shape, we definitely can't fail the assertFalse
            self.assertFalse(np.allclose(induc_after, induc_reset))

        for before, after in zip(variational_params_before, variational_params_after):
            self.assertFalse(np.allclose(before, after))

        for after, reset in zip(variational_params_after, variational_params_reset):
            if after.shape == reset.shape:  # Same as above
                self.assertFalse(np.allclose(after, reset))

    def test_predict_p(self):
        """
        Verify analytic p-space mean and var is correct.
        """
        X, y = self.X, self.y
        inducing_size = 10

        model = GPClassificationModel(
            dim=1,
            inducing_size=inducing_size,
        )
        model.fit(X, y)

        pmean_analytic, pvar_analytic = model.predict_probability(X)

        fsamps = model.sample(X, 150000)
        psamps = norm.cdf(fsamps)
        pmean_samp = psamps.mean(0)
        pvar_samp = psamps.var(0)
        # TODO these tolerances are a bit loose, verify this is right.
        self.assertTrue(np.allclose(pmean_analytic, pmean_samp, atol=0.01))
        self.assertTrue(np.allclose(pvar_analytic, pvar_samp, atol=0.01))


class GPClassificationTest(unittest.TestCase):
    def setUp(self):
        self.seed = 1
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def test_1d_single_probit_new_interface(self):
        n_init = 50
        n_opt = 1
        lb = torch.tensor([-4.0])
        ub = torch.tensor([4.0])
        inducing_size = 10

        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=self.seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(
                    dim=1,
                    inducing_size=inducing_size,
                ),
                generator=OptimizeAcqfGenerator(
                    acqf=qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}, lb=lb, ub=ub
                ),
                min_asks=n_opt,
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(model_list)

        while not strat.finished:
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(f_1d(next_x))])

        self.assertTrue(strat.y.shape[0] == n_init + n_opt)

        x = torch.linspace(-4, 4, 100)

        zhat, _ = strat.predict(x)

        # true max is 0, very loose test
        self.assertTrue(np.abs(x[np.argmax(zhat.detach().numpy())]) < 0.5)

    def test_1d_single_probit_batched(self):
        n_init = 50
        n_opt = 2
        lb = torch.tensor([-4.0])
        ub = torch.tensor([4.0])
        inducing_size = 10

        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=self.seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(
                    dim=1,
                    inducing_size=inducing_size,
                ),
                generator=OptimizeAcqfGenerator(
                    acqf=qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}, lb=lb, ub=ub
                ),
                min_asks=n_opt,
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(model_list)
        while not strat.finished:
            next_x = strat.gen(num_points=2)
            strat.add_data(next_x, bernoulli.rvs(f_1d(next_x)).squeeze())

        self.assertEqual(strat.y.shape[0], n_init + n_opt)

        x = torch.linspace(-4, 4, 100)

        zhat, _ = strat.predict(x)

        # true max is 0, very loose test
        self.assertTrue(np.abs(x[np.argmax(zhat.detach().numpy())]) < 0.5)

    def test_1d_single_probit(self):
        n_init = 50
        n_opt = 1
        lb = torch.tensor([-4.0])
        ub = torch.tensor([4.0])
        inducing_size = 10

        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=self.seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(
                    dim=1,
                    inducing_size=inducing_size,
                ),
                generator=OptimizeAcqfGenerator(
                    acqf=qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}, lb=lb, ub=ub
                ),
                min_asks=n_opt,
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(model_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(f_1d(next_x))])

        x = torch.linspace(-4, 4, 100)

        zhat, _ = strat.predict(x)

        # true max is 0, very loose test
        self.assertTrue(np.abs(x[np.argmax(zhat.detach().numpy())]) < 0.5)

    def test_1d_single_probit_pure_exploration(self):
        n_init = 50
        n_opt = 1
        lb = torch.tensor([-4.0])
        ub = torch.tensor([4.0])
        inducing_size = 10

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=self.seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(
                    dim=1,
                    inducing_size=inducing_size,
                ),
                generator=OptimizeAcqfGenerator(
                    acqf=qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}, lb=lb, ub=ub
                ),
                min_asks=n_opt,
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(strat_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(norm.cdf(next_x))])

        x = torch.linspace(-4, 4, 100)

        zhat, _ = strat.predict(x)

        # f(x) = x so we're just looking at corr between cdf(zhat) and cdf(x)
        self.assertTrue(
            pearsonr(norm.cdf(zhat.detach().numpy()).flatten(), norm.cdf(x).flatten())[
                0
            ]
            > 0.95
        )

    def test_2d_single_probit_pure_exploration(self):
        n_init = 50
        n_opt = 1
        lb = torch.tensor([-1.0, -1.0])
        ub = torch.tensor([1.0, 1.0])
        inducing_size = 10

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=self.seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(
                    dim=2,
                    inducing_size=inducing_size,
                ),
                generator=OptimizeAcqfGenerator(
                    acqf=qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}, lb=lb, ub=ub
                ),
                min_asks=n_opt,
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(strat_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen().detach().cpu().numpy()
            strat.add_data(next_x, [bernoulli.rvs(cdf_new_novel_det(next_x))])

        xy = np.mgrid[-1:1:30j, -1:1:30j].reshape(2, -1).T
        post_mean, _ = strat.predict(torch.Tensor(xy))
        phi_post_mean = norm.cdf(post_mean.reshape(30, 30).detach().numpy())

        phi_post_true = cdf_new_novel_det(xy)

        self.assertTrue(
            pearsonr(phi_post_mean.flatten(), phi_post_true.flatten())[0] > 0.75
        )

    def test_1d_single_targeting(self):
        n_init = 50
        n_opt = 1
        lb = torch.tensor([-4.0])
        ub = torch.tensor([4.0])
        inducing_size = 10

        target = 0.75

        def obj(x):
            return -((Normal(0, 1).cdf(x[..., 0]) - target) ** 2)

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=self.seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(
                    dim=1,
                    inducing_size=inducing_size,
                ),
                generator=OptimizeAcqfGenerator(
                    acqf=qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}, lb=lb, ub=ub
                ),
                min_asks=n_opt,
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(strat_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(norm.cdf(next_x))])

        x = torch.linspace(-4, 4, 100)

        zhat, _ = strat.predict(x)

        # since target is 0.75, find the point at which f_est is 0.75
        est_max = x[np.argmin((norm.cdf(zhat.detach().numpy()) - 0.75) ** 2)]
        # since true z is just x, the true max is where phi(x)=0.75,
        self.assertTrue(np.abs(est_max - norm.ppf(0.75)) < 0.5)

    def test_1d_jnd(self):
        n_init = 150
        n_opt = 1
        lb = torch.tensor([-4.0])
        ub = torch.tensor([4.0])
        inducing_size = 10

        target = 0.5

        def obj(x):
            return -((Normal(0, 1).cdf(x[..., 0]) - target) ** 2)

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=self.seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(
                    dim=1,
                    inducing_size=inducing_size,
                ),
                generator=OptimizeAcqfGenerator(
                    acqf=qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}, lb=lb, ub=ub
                ),
                min_asks=n_opt,
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(strat_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, Normal(0, 1).cdf(next_x / 1.5).bernoulli().view(-1))

        x = torch.linspace(-4, 4, 100)

        zhat, _ = strat.predict(x)

        # we expect jnd close to the target to be close to the correct
        # jnd (1.5), and since this is linear model this should be true
        # for both definitions of JND
        jnd_step = get_jnd(
            strat.model,
            strat.lb,
            strat.ub,
            dim=strat.dim,
            grid=x[:, None],
            method="step",
        )
        est_jnd_step = jnd_step[50]
        # looser test because step-jnd is hurt more by reverting to the mean
        self.assertTrue(np.abs(est_jnd_step - 1.5) < 0.5)

        jnd_taylor = get_jnd(
            strat.model,
            strat.lb,
            strat.ub,
            dim=strat.dim,
            grid=x[:, None],
            method="taylor",
        )
        est_jnd_taylor = jnd_taylor[50]
        self.assertTrue(np.abs(est_jnd_taylor - 1.5) < 0.25)

    def test_1d_single_lse(self):
        n_init = 50
        n_opt = 1
        lb = torch.tensor([-4.0])
        ub = torch.tensor([4.0])
        inducing_size = 10

        # target is in z space not phi(z) space, maybe that's
        # weird
        extra_acqf_args = {"target": 0.75, "beta": 1.96}

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=self.seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(
                    dim=1,
                    inducing_size=inducing_size,
                ),
                min_asks=n_opt,
                generator=OptimizeAcqfGenerator(
                    acqf=MCLevelSetEstimation, acqf_kwargs=extra_acqf_args, lb=lb, ub=ub
                ),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(strat_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(norm.cdf(next_x))])

        x = torch.linspace(-4, 4, 100)

        zhat, _ = strat.predict(x)
        # since target is 0.75, find the point at which f_est is 0.75
        est_max = x[np.argmin((norm.cdf(zhat.detach().cpu().numpy()) - 0.75) ** 2)]
        # since true z is just x, the true max is where phi(x)=0.75,
        self.assertTrue(np.abs(est_max - norm.ppf(0.75)) < 0.5)

    def test_2d_single_probit(self):
        n_init = 150
        n_opt = 1
        lb = torch.tensor([-1.0, -1.0])
        ub = torch.tensor([1.0, 1.0])
        inducing_size = 20

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=self.seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(
                    dim=2,
                    inducing_size=inducing_size,
                ),
                generator=OptimizeAcqfGenerator(
                    acqf=qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}, lb=lb, ub=ub
                ),
                min_asks=n_opt,
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(strat_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(f_2d(next_x[None, :]))])

        xy = np.mgrid[-1:1:30j, -1:1:30j].reshape(2, -1).T
        zhat, _ = strat.predict(torch.Tensor(xy))

        self.assertTrue(np.all(np.abs(xy[np.argmax(zhat.detach().numpy())]) < 0.5))

    def test_extra_ask_warns(self):
        n_init = 3
        n_opt = 1
        lb = torch.tensor([-4.0])
        ub = torch.tensor([4.0])
        inducing_size = 10

        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=self.seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(
                    dim=1,
                    inducing_size=inducing_size,
                ),
                generator=OptimizeAcqfGenerator(
                    acqf=qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}, lb=lb, ub=ub
                ),
                min_asks=n_opt,
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(model_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(norm.cdf(f_1d(next_x)))])

        with self.assertLogs() as log:
            strat.gen()

        self.assertIn("Ran out of generators", log[0][0].message)

    def test_hyperparam_consistency(self):
        # verify that creating the model `from_config` or with `__init__` has the same hyperparams
        m1 = GPClassificationModel(
            dim=2,
            inducing_size=2,
        )

        config = Config(
            config_dict={
                "common": {
                    "parnames": ["par1", "par2"],
                    "lb": "[1, 2]",
                    "ub": "[3, 4]",
                    "inducing_size": 2,
                },
                "par1": {"value_type": "float"},
                "par2": {"value_type": "float"},
            }
        )
        m2 = GPClassificationModel.from_config(config=config)
        self.assertTrue(isinstance(m1.covar_module, type(m2.covar_module)))
        self.assertTrue(isinstance(m1.covar_module, type(m2.covar_module)))
        self.assertTrue(isinstance(m1.mean_module, type(m2.mean_module)))
        m1priors = list(m1.covar_module.named_priors())
        m2priors = list(m2.covar_module.named_priors())
        for p1, p2 in zip(m1priors, m2priors):
            name1, parent1, prior1, paramtransforms1, priortransforms1 = p1
            name2, parent2, prior2, paramtransforms2, priortransforms2 = p2
            self.assertTrue(name1 == name2)
            self.assertTrue(isinstance(parent1, type(parent2)))
            self.assertTrue(isinstance(prior1, type(prior2)))
            # no obvious way to test paramtransform equivalence)

        self.assertTrue(m1.inducing_size == m2.inducing_size)

    def test_pairwise_factory(self):
        config_str = """
        [common]
        parnames = [foo1, bar1, foo2, bar2]
        stimuli_per_trial = 1

        [foo1]
        par_type = continuous
        lower_bound = 0.3
        upper_bound = 0.9

        [bar1]
        par_type = continuous
        lower_bound = 24
        upper_bound = 66

        [foo2]
        par_type = continuous
        lower_bound = 0.3
        upper_bound = 0.9

        [bar2]
        par_type = continuous
        lower_bound = 24
        upper_bound = 66

        [GPClassificationModel]
        mean_covar_factory = pairwise_mean_covar_factory
        """
        config = Config(config_str=config_str)
        m = GPClassificationModel.from_config(config=config)
        self.assertIsInstance(m.mean_module, ZeroMean)
        self.assertIsInstance(m.covar_module, PairwiseKernel)

    def test_constraint_model(self):
        x = torch.linspace(0, 1, 50).unsqueeze(-1)
        y_true = f_1d(x, mu=0.75)
        y = torch.tensor(bernoulli.rvs(y_true))

        model = GPClassificationModel(
            dim=1,
            constraint_locations=torch.tensor([[0.0], [1.0]]),
            constraint_values=torch.tensor([y_true[0], y_true[-1]]),
            constraint_strengths=torch.tensor([1e-6, 1e-6]),
            inducing_size=10,
        )

        model.fit(x, y)

        y_pred, y_var = model.predict_probability(x)

        self.assertTrue(
            torch.allclose(y_pred[0], y_true[0], atol=1e-4, rtol=1e-4), y_pred
        )
        self.assertTrue(torch.allclose(y_pred[-1], y_true[-1], atol=1e-4, rtol=1e-4))

        # Y_var should be small at the constraint locations
        self.assertTrue(y_var[0] < 1e-4)
        self.assertTrue(y_var[-1] < 1e-4)

    def test_auto_constraint_strength(self):
        x = torch.linspace(0, 1, 50).unsqueeze(-1)
        y_true = f_1d(x, mu=0.75)
        y = torch.tensor(bernoulli.rvs(y_true))

        model = GPClassificationModel(
            dim=1,
            constraint_locations=torch.tensor([[0.0], [1.0]]),
            constraint_values=torch.tensor([y_true[0], y_true[-1]]),
        )

        # Smoke test
        model.fit(x, y)

        self.assertLess(model.constraint_strengths[0], 0.2)
        self.assertLess(model.constraint_strengths[1], 0.2)

    def test_constraint_strat(self):
        config_str = """
            [common]
            parnames = [par1, par2]
            strategy_names = [init_strat, opt_strat]
            stimuli_per_trial = 1
            outcome_types = [binary]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 500

            [par2]
            par_type = continuous
            lower_bound = 1
            upper_bound = 2

            [init_strat]
            min_asks = 50
            generator = SobolGenerator

            [opt_strat]
            min_asks = 2
            model = GPClassificationModel
            generator = OptimizeAcqfGenerator
            refit_every = 2 # Checks warmstart

            [GPClassificationModel]
            constraint_locations = [0, 1]
            constraint_values = [0.5]
            constraint_strengths = [1e-4]

            [OptimizeAcqfGenerator]
            acqf = qLogNoisyExpectedImprovement
        """
        config = Config(config_str=config_str)
        strat = SequentialStrategy.from_config(config=config)

        while not strat.finished:
            x = strat.gen()
            prob = f_2d_target(
                strat.transforms.transform(x), target=torch.tensor([0.5, 0.5])
            )
            response = bernoulli.rvs(prob)
            strat.add_data(x, [response])

        # These should be in transformed space and latent space
        model = strat.model
        self.assertTrue(
            torch.all(model.constraint_locations == torch.tensor([[0.0, 0.0]]))
        )
        self.assertTrue(torch.all(model.constraint_values == torch.tensor([[0.0]])))

        pred_y, pred_var = strat.predict(
            torch.tensor([[0.0, 1.0]]), probability_space=True
        )
        self.assertLess(pred_y - 0.5, 1e-3)
        self.assertLess(pred_var, 1e-4)

    def test_multi_constraint_strat(self):
        config_str = """
            [common]
            parnames = [par1, par2]
            strategy_names = [init_strat, opt_strat]
            stimuli_per_trial = 1
            outcome_types = [binary]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 500

            [par2]
            par_type = continuous
            lower_bound = 1
            upper_bound = 2

            [init_strat]
            min_asks = 1
            generator = SobolGenerator

            [opt_strat]
            min_asks = 1
            model = GPClassificationModel
            generator = OptimizeAcqfGenerator
            refit_every = 2 # Checks warmstart

            [GPClassificationModel]
            constraint_locations = [[0, 1], [0, 2]]
            constraint_values = [0.5, 0.5]
            constraint_strengths = [1e-4, 1e-4]

            [OptimizeAcqfGenerator]
            acqf = qLogNoisyExpectedImprovement
        """
        config = Config(config_str=config_str)
        strat = SequentialStrategy.from_config(config=config)

        # Smoketest
        while not strat.finished:
            x = strat.gen()
            prob = f_2d_target(
                strat.transforms.transform(x), target=torch.tensor([0.5, 0.5])
            )
            response = bernoulli.rvs(prob)
            strat.add_data(x, [response])

        # These should be in transformed space and latent space
        model = strat.model
        self.assertTrue(
            torch.all(model.constraint_locations == torch.tensor([[0, 0], [0, 1]]))
        )
        self.assertTrue(torch.all(model.constraint_values == torch.tensor([0.0, 0.0])))

    def test_constraint_factory(self):
        config_str = """
            [common]
            parnames = [par1, par2]
            strategy_names = [init_strat, opt_strat]
            stimuli_per_trial = 1
            outcome_types = [binary]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 500

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 2

            [init_strat]
            min_asks = 10
            generator = SobolGenerator

            [opt_strat]
            min_asks = 1
            model = GPClassificationModel
            generator = OptimizeAcqfGenerator

            [GPClassificationModel]
            constraint_factory = constraint_factory
            inducing_size = 10

            [constraint_factory]
            constraint_lower = [[0, 0], [0, 0]]
            constraint_upper = [[0, 2], [500, 0]]
            constraint_values = [0.5, 0.5]
            constraint_strengths = [1e-2, 1e-2]
            points_per_dim = 3

            [OptimizeAcqfGenerator]
            acqf = qLogNoisyExpectedImprovement
        """
        config = Config(config_str=config_str)
        strat = SequentialStrategy.from_config(config=config)

        while not strat.finished:
            x = strat.gen()
            prob = f_2d_target(
                strat.transforms.transform(x), target=torch.tensor([0.5, 0.5])
            )
            response = bernoulli.rvs(prob)
            strat.add_data(x, [response])

        # These should be in transformed space and latent space
        model = strat.model

        # Check if enough constraints were made, 3 cause 0, 0 is duplicated
        self.assertEqual(model.constraint_locations.shape[0], 5)
        self.assertEqual(model.constraint_values.shape[0], 5)
        self.assertEqual(model.constraint_strengths.shape[0], 5)

        # Check if all constraint_locations are in range
        for loc in model.constraint_locations:
            self.assertTrue(torch.all(loc >= 0))
            self.assertTrue(torch.all(loc <= 1))
            self.assertTrue(torch.any(loc == 0))

        # Check if all constraint_values the expected value
        self.assertTrue(torch.all(model.constraint_values == torch.tensor([[0.0]])))

        pred_x_dim1 = torch.linspace(0, 500, 10)
        pred_x_dim1 = torch.vstack([pred_x_dim1, torch.zeros_like(pred_x_dim1)]).T
        pred_x_dim2 = torch.linspace(0, 2, 10)
        pred_x_dim2 = torch.vstack([torch.zeros_like(pred_x_dim2), pred_x_dim2]).T
        pred_x = torch.cat([pred_x_dim1, pred_x_dim2])
        pred_y, pred_var = strat.predict(pred_x, probability_space=True)
        self.assertTrue(torch.all(pred_y - 0.5 < 1e-2), pred_y)
        self.assertTrue(torch.all(pred_var < 1e-2), pred_var)

    def test_constraint_factory_zero_points(self):
        config_str = """
            [common]
            parnames = [par1, par2]
            strategy_names = [init_strat, opt_strat]
            stimuli_per_trial = 1
            outcome_types = [binary]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 500

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 2

            [init_strat]
            min_asks = 1
            generator = SobolGenerator

            [opt_strat]
            min_asks = 1
            model = GPClassificationModel
            generator = OptimizeAcqfGenerator

            [GPClassificationModel]
            constraint_factory = constraint_factory

            [constraint_factory]
            points_per_dim = 0
            constraint_lower = [[0, 0], [0, 0]]
            constraint_upper = [[0, 2], [500, 0]]
            constraint_values = [0.5, 0.5]
            constraint_strengths = [1e-4, 1e-4]

            [OptimizeAcqfGenerator]
            acqf = qLogNoisyExpectedImprovement
        """
        config = Config(config_str=config_str)
        strat = SequentialStrategy.from_config(config=config)

        # Verify that no constraints were created
        model = strat.strat_list[-1].model
        self.assertIsNone(model.constraint_locations)
        self.assertIsNone(model.constraint_values)
        self.assertIsNone(model.constraint_strengths)

    def test_constraint_combine_one(self):
        """Test that constraints from constraint_factory are combined with one explicit constraints in a strategy."""
        config_str = """
            [common]
            parnames = [par1, par2]
            strategy_names = [init_strat, opt_strat]
            stimuli_per_trial = 1
            outcome_types = [binary]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 500

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 2

            [init_strat]
            min_asks = 1
            generator = SobolGenerator

            [opt_strat]
            min_asks = 1
            model = GPClassificationModel
            generator = OptimizeAcqfGenerator

            [GPClassificationModel]
            constraint_locations = [100, 0.4]
            constraint_values = [1]
            constraint_strengths = [1e-3]
            constraint_factory = constraint_factory

            [constraint_factory]
            constraint_lower = [[0, 0], [0, 0]]
            constraint_upper = [[0, 2], [500, 0]]
            constraint_values = [0.5, 0.5]
            constraint_strengths = [1e-2, 1e-2]
            points_per_dim = 2

            [OptimizeAcqfGenerator]
            acqf = qLogNoisyExpectedImprovement
        """
        config = Config(config_str=config_str)
        strat = SequentialStrategy.from_config(config=config)

        model = strat.strat_list[-1].model

        # Check that the explicit constraints are present, should be first
        self.assertTrue(
            torch.all(model.constraint_locations[0] == torch.tensor([[0.2, 0.2]]))
        )
        self.assertTrue(  # Some big number since this is latent space
            torch.all(model.constraint_values[0] > torch.tensor([4.0]))
        )
        self.assertTrue(model.constraint_values[0] != torch.inf)

        self.assertTrue(
            torch.all(model.constraint_strengths[0] == torch.tensor([1e-3]))
        )

        factory_locations = model.constraint_locations[1:]
        factory_values = model.constraint_values[1:]
        factory_strengths = model.constraint_strengths[1:]

        self.assertTrue(torch.any(factory_locations == 0.0, dim=1).all())
        self.assertEqual(factory_locations.shape[0], 3)
        self.assertTrue(torch.all(factory_values == 0))
        self.assertEqual(factory_values.shape[0], 3)
        self.assertTrue(torch.all(factory_strengths == 1e-2))
        self.assertEqual(factory_strengths.shape[0], 3)

    def test_constraint_combine_multiple(self):
        """Test that constraints from constraint_factory are combined with multiple explicit constraints in a strategy."""
        config_str = """
            [common]
            parnames = [par1, par2]
            strategy_names = [init_strat, opt_strat]
            stimuli_per_trial = 1
            outcome_types = [binary]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 500

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 2

            [init_strat]
            min_asks = 1
            generator = SobolGenerator

            [opt_strat]
            min_asks = 1
            model = GPClassificationModel
            generator = OptimizeAcqfGenerator

            [GPClassificationModel]
            constraint_locations = [[100, 0.4], [200, 1.0], [300, 1.5]]
            constraint_values = [1, 0.7, 0.3]
            constraint_strengths = [1e-3, 2e-3, 3e-3]
            constraint_factory = constraint_factory

            [constraint_factory]
            constraint_lower = [[0, 0], [0, 0]]
            constraint_upper = [[0, 2], [500, 0]]
            constraint_values = [0.5, 0.5]
            constraint_strengths = [1e-2, 1e-2]
            points_per_dim = 2

            [OptimizeAcqfGenerator]
            acqf = qLogNoisyExpectedImprovement
        """
        config = Config(config_str=config_str)
        strat = SequentialStrategy.from_config(config=config)

        model = strat.strat_list[-1].model

        # Check that the explicit constraints are present
        expected_locations = torch.tensor([[0.2, 0.2], [0.4, 0.5], [0.6, 0.75]])
        expected_strengths = torch.tensor([1e-3, 2e-3, 3e-3])

        # Get actual values from model for the first 3 constraints
        actual_locations = model.constraint_locations[:3]
        actual_strengths = model.constraint_strengths[:3]
        actual_values = model.constraint_values[:3]

        self.assertTrue(torch.all(torch.eq(actual_locations, expected_locations)))
        self.assertTrue(torch.all(torch.eq(actual_strengths, expected_strengths)))

        # Check values with specific conditions
        self.assertTrue(actual_values[0] > 4.0 and actual_values[0] != torch.inf)
        self.assertTrue(actual_values[1] > 0.0)
        self.assertTrue(actual_values[2] < 0.0)

        # Check factory constraints
        factory_locations = model.constraint_locations[3:]
        factory_values = model.constraint_values[3:]
        factory_strengths = model.constraint_strengths[3:]

        self.assertTrue(torch.any(factory_locations == 0.0, dim=1).all())
        self.assertEqual(factory_locations.shape[0], 3)
        self.assertTrue(torch.all(factory_values == 0))
        self.assertEqual(factory_values.shape[0], 3)
        self.assertTrue(torch.all(factory_strengths == 1e-2))
        self.assertEqual(factory_strengths.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
