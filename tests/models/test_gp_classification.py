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

from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
from aepsych.acquisition import MCLevelSetEstimation
from aepsych.config import Config
from aepsych.generators import OptimizeAcqfGenerator, SobolGenerator
from aepsych.models import GPClassificationModel
from aepsych.strategy import SequentialStrategy, Strategy
from botorch.acquisition import qUpperConfidenceBound
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from scipy.stats import bernoulli, norm, pearsonr
from sklearn.datasets import make_classification
from torch.distributions import Normal
from functools import partial
from torch.optim import Adam

from ..common import cdf_new_novel_det, f_1d, f_2d


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
        model = GPClassificationModel(
            torch.Tensor([-3]), torch.Tensor([3]), inducing_size=10
        )

        model.fit(X[:50], y[:50])

        # pspace
        pm, _ = model.predict(X[:50], probability_space=True)
        pred = (pm > 0.5).numpy()
        npt.assert_allclose(pred, y[:50])

        # fspace
        pm, _ = model.predict(X[:50], probability_space=False)
        pred = (pm > 0).numpy()
        npt.assert_allclose(pred, y[:50])

        # smoke test update
        model.update(X, y)

        # pspace
        pm, _ = model.predict(X, probability_space=True)
        pred = (pm > 0.5).numpy()
        npt.assert_allclose(pred, y)

        # fspace
        pm, _ = model.predict(X, probability_space=False)
        pred = (pm > 0).numpy()
        npt.assert_allclose(pred, y)

    def test_1d_classification_pytorchopt(self):
        """
        Just see if we memorize the training set
        """
        X, y = self.X, self.y
        model = GPClassificationModel(
            torch.Tensor([-3]), torch.Tensor([3]), inducing_size=10
        )

        model.fit(
            X[:50],
            y[:50],
            optimizer=fit_gpytorch_mll_torch,
            optimizer_kwargs={"stopping_criterion":ExpMAStoppingCriterion(maxiter=30),'optimizer':partial(Adam, lr=0.05)}
        )

        # pspace
        pm, _ = model.predict(X[:50], probability_space=True)
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
            optimizer_kwargs={"stopping_criterion":ExpMAStoppingCriterion(maxiter=30)}
        )

        # pspace
        pm, _ = model.predict(X, probability_space=True)
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
        lb = [-3000, -0.003]
        ub = [3000, 0.003]

        model = GPClassificationModel(lb=lb, ub=ub, inducing_size=20)

        model.fit(X[:50], y[:50])

        # pspace
        pm, _ = model.predict(X[:50], probability_space=True)
        pred = (pm > 0.5).numpy()
        npt.assert_allclose(pred, y[:50])

        # fspace
        pm, _ = model.predict(X[:50], probability_space=False)
        pred = (pm > 0).numpy()
        npt.assert_allclose(pred, y[:50])

        # smoke test update
        model.update(X, y)

        # pspace
        pm, _ = model.predict(X, probability_space=True)
        pred = (pm > 0.5).numpy()
        npt.assert_allclose(pred, y)

        # fspace
        pm, _ = model.predict(X, probability_space=False)
        pred = (pm > 0).numpy()
        npt.assert_allclose(pred, y)

    def test_reset_hyperparams(self):
        model = GPClassificationModel(lb=[-3], ub=[3], inducing_size=20)

        os_before = model.covar_module.outputscale.clone().detach().numpy()
        ls_before = model.covar_module.base_kernel.lengthscale.clone().detach().numpy()
        model.fit(torch.Tensor(self.X), torch.Tensor(self.y))

        os_after = model.covar_module.outputscale.clone().detach().numpy()
        ls_after = model.covar_module.base_kernel.lengthscale.clone().detach().numpy()

        model._reset_hyperparameters()

        os_reset = model.covar_module.outputscale.clone().detach().numpy()
        ls_reset = model.covar_module.base_kernel.lengthscale.clone().detach().numpy()

        # before should be different from after and after should be different
        # from reset but before and reset should be same
        self.assertFalse(np.allclose(os_before, os_after))
        self.assertFalse(np.allclose(os_after, os_reset))
        self.assertTrue(np.allclose(os_before, os_reset))
        self.assertFalse(np.allclose(ls_before, ls_after))
        self.assertFalse(np.allclose(ls_after, ls_reset))
        self.assertTrue(np.allclose(ls_before, ls_reset))

    def test_reset_variational_strategy(self):

        model = GPClassificationModel(lb=[-3], ub=[3], inducing_size=20)

        variational_params_before = [
            v.clone().detach().numpy() for v in model.variational_parameters()
        ]
        induc_before = model.variational_strategy.inducing_points

        model.fit(torch.Tensor(self.X), torch.Tensor(self.y))

        variational_params_after = [
            v.clone().detach().numpy() for v in model.variational_parameters()
        ]
        induc_after = model.variational_strategy.inducing_points

        model._reset_variational_strategy()

        variational_params_reset = [
            v.clone().detach().numpy() for v in model.variational_parameters()
        ]
        induc_reset = model.variational_strategy.inducing_points

        # before should be different from after and after should be different
        # from reset
        self.assertFalse(np.allclose(induc_before, induc_after))
        self.assertFalse(np.allclose(induc_after, induc_reset))
        for before, after in zip(variational_params_before, variational_params_after):
            self.assertFalse(np.allclose(before, after))

        for after, reset in zip(variational_params_after, variational_params_reset):
            self.assertFalse(np.allclose(after, reset))

    def test_predict_p(self):
        """
        Verify analytic p-space mean and var is correct.
        """
        X, y = self.X, self.y
        model = GPClassificationModel(
            torch.Tensor([-3]), torch.Tensor([3]), inducing_size=10
        )
        model.fit(X, y)

        pmean_analytic, pvar_analytic = model.predict(X, probability_space=True)

        fsamps = model.sample(X, 150000)
        psamps = norm.cdf(fsamps)
        pmean_samp = psamps.mean(0)
        pvar_samp = psamps.var(0)
        # TODO these tolerances are a bit loose, verify this is right.
        self.assertTrue(np.allclose(pmean_analytic, pmean_samp, atol=0.001))
        self.assertTrue(np.allclose(pvar_analytic, pvar_samp, atol=0.001))


class GPClassificationTest(unittest.TestCase):
    def test_1d_single_probit_new_interface(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        lb = -4.0
        ub = 4.0

        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub, inducing_size=10),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
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

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 2
        lb = -4.0
        ub = 4.0

        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub, inducing_size=10),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
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

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        lb = -4.0
        ub = 4.0

        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub, inducing_size=10),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
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

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        lb = -4.0
        ub = 4.0

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub, inducing_size=10),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
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
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        lb = [-1, -1]
        ub = [1, 1]

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub, inducing_size=10),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
                ),
                min_asks=n_opt,
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(strat_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(cdf_new_novel_det(next_x))])

        xy = np.mgrid[-1:1:30j, -1:1:30j].reshape(2, -1).T
        post_mean, _ = strat.predict(torch.Tensor(xy))
        phi_post_mean = norm.cdf(post_mean.reshape(30, 30).detach().numpy())

        phi_post_true = cdf_new_novel_det(xy)

        self.assertTrue(
            pearsonr(phi_post_mean.flatten(), phi_post_true.flatten())[0] > 0.9
        )

    def test_1d_single_targeting(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        lb = -4.0
        ub = 4.0

        target = 0.75

        def obj(x):
            return -((Normal(0, 1).cdf(x[..., 0]) - target) ** 2)

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub, inducing_size=10),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
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
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 150
        n_opt = 1
        lb = -4.0
        ub = 4.0

        target = 0.5

        def obj(x):
            return -((Normal(0, 1).cdf(x[..., 0]) - target) ** 2)

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub, inducing_size=10),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
                ),
                min_asks=n_opt,
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(strat_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(norm.cdf(next_x / 1.5))])

        x = torch.linspace(-4, 4, 100)

        zhat, _ = strat.predict(x)

        # we expect jnd close to the target to be close to the correct
        # jnd (1.5), and since this is linear model this should be true
        # for both definitions of JND
        jnd_step = strat.get_jnd(grid=x[:, None], method="step")
        est_jnd_step = jnd_step[50]
        # looser test because step-jnd is hurt more by reverting to the mean
        self.assertTrue(np.abs(est_jnd_step - 1.5) < 0.5)

        jnd_taylor = strat.get_jnd(grid=x[:, None], method="taylor")
        est_jnd_taylor = jnd_taylor[50]
        self.assertTrue(np.abs(est_jnd_taylor - 1.5) < 0.25)

    def test_1d_single_lse(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        lb = -4.0
        ub = 4.0

        # target is in z space not phi(z) space, maybe that's
        # weird
        extra_acqf_args = {"target": 0.75, "beta": 1.96}

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub, inducing_size=10),
                min_asks=n_opt,
                generator=OptimizeAcqfGenerator(
                    MCLevelSetEstimation, acqf_kwargs=extra_acqf_args
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
        est_max = x[np.argmin((norm.cdf(zhat.detach().numpy()) - 0.75) ** 2)]
        # since true z is just x, the true max is where phi(x)=0.75,
        self.assertTrue(np.abs(est_max - norm.ppf(0.75)) < 0.5)

    def test_2d_single_probit(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 150
        n_opt = 1
        lb = [-1, -1]
        ub = [1, 1]

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub, inducing_size=20),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
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
        # test that when we ask more times than we have models, we warn but keep going
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 3
        n_opt = 1
        lb = -4.0
        ub = 4.0

        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub, inducing_size=10),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
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

        with self.assertWarns(RuntimeWarning):
            strat.gen()

    def test_1d_query(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        lb = -4.0
        ub = 4.0

        strat = Strategy(
            lb=lb,
            ub=ub,
            min_asks=1,
            generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
            model=GPClassificationModel(lb=lb, ub=ub, inducing_size=50),
            stimuli_per_trial=1,
            outcome_types=["binary"],
        )

        # mock the posterior call and remove calls that don't need
        # to happen
        def get_fake_posterior(X, posterior_transform=None):
            fmean = torch.sin(torch.pi * X / 4).squeeze(-1)
            fcov = torch.eye(fmean.shape[0])
            fake_posterior = GPyTorchPosterior(
                mvn=MultivariateNormal(mean=fmean, covariance_matrix=fcov)
            )
            return fake_posterior

        strat.model.posterior = get_fake_posterior
        strat.model.__call__ = MagicMock()
        strat.model.fit = MagicMock()

        x = strat.gen(1)
        y = torch.Tensor([1])
        strat.add_data(x, y)
        strat.model.set_train_data(x, y)
        # We expect the global max to be at (2, 1), the min at (-2, -1)
        fmax, argmax = strat.get_max()
        self.assertTrue(np.allclose(fmax, 1))
        self.assertTrue(np.allclose(argmax, 2))

        fmin, argmin = strat.get_min()
        self.assertTrue(np.allclose(fmin, -1))
        self.assertTrue(np.allclose(argmin, -2, atol=0.2))

        # Inverse query at val .85 should return (.85,[2.7])
        val, loc = strat.inv_query(0.85, constraints={})
        self.assertTrue(np.allclose(val, 0.85))
        self.assertTrue(np.allclose(loc.item(), 2.7, atol=1e-2))

    def test_hyperparam_consistency(self):
        # verify that creating the model `from_config` or with `__init__` has the same hyperparams

        m1 = GPClassificationModel(lb=[1, 2], ub=[3, 4])

        m2 = GPClassificationModel.from_config(
            config=Config(config_dict={"common": {"lb": "[1,2]", "ub": "[3,4]"}})
        )
        self.assertTrue(isinstance(m1.covar_module, type(m2.covar_module)))
        self.assertTrue(
            isinstance(m1.covar_module.base_kernel, type(m2.covar_module.base_kernel))
        )
        self.assertTrue(isinstance(m1.mean_module, type(m2.mean_module)))
        m1priors = list(m1.covar_module.named_priors())
        m2priors = list(m2.covar_module.named_priors())
        for p1, p2 in zip(m1priors, m2priors):
            name1, parent1, prior1, paramtransforms1, priortransforms1 = p1
            name2, parent2, prior2, paramtransforms2, priortransforms2 = p2
            self.assertTrue(name1 == name2)
            self.assertTrue(isinstance(parent1, type(parent2)))
            self.assertTrue(isinstance(prior1, type(prior2)))
            # no obvious way to test paramtransform equivalence


if __name__ == "__main__":
    unittest.main()
