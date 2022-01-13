#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import numpy as np
import numpy.testing as npt
import torch
from aepsych.acquisition.lse import LevelSetEstimation
from aepsych.config import Config
from aepsych.generators import OptimizeAcqfGenerator, SobolGenerator
from aepsych.models import GPClassificationModel
from aepsych.strategy import SequentialStrategy, Strategy
from botorch.acquisition import qUpperConfidenceBound
from scipy.stats import bernoulli, norm, pearsonr
from sklearn.datasets import make_classification
from torch.distributions import Normal
from botorch.acquisition.objective import GenericMCObjective

from .common import cdf_new_novel_det, f_1d, f_2d


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
                n_trials=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
                ),
                n_trials=n_opt,
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
                n_trials=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
                ),
                n_trials=n_opt,
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
                n_trials=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
                ),
                n_trials=n_opt,
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
                n_trials=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
                ),
                n_trials=n_opt,
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
                n_trials=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
                ),
                n_trials=n_opt,
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
                n_trials=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
                ),
                n_trials=n_opt,
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
                n_trials=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
                ),
                n_trials=n_opt,
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
        extra_acqf_args = {"target": norm.ppf(0.75), "beta": 1.96}

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                n_trials=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub),
                n_trials=n_opt,
                generator=OptimizeAcqfGenerator(
                    LevelSetEstimation, acqf_kwargs=extra_acqf_args
                ),
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
                n_trials=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
                ),
                n_trials=n_opt,
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
                n_trials=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound, acqf_kwargs={"beta": 1.96}
                ),
                n_trials=n_opt,
            ),
        ]

        strat = SequentialStrategy(model_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(norm.cdf(f_1d(next_x)))])

        with self.assertWarns(RuntimeWarning):
            strat.gen()

    def test_gpclassification_config(self):
        config_str = """
        [common]
        lb = [0, 0]
        ub = [1, 1]
        outcome_type = single_probit
        parnames = [par1, par2]
        strategy_names = [init_strat, opt_strat]
        acqf = LevelSetEstimation
        model = GPClassificationModel

        [init_strat]
        n_trials = 10
        generator = SobolGenerator

        [opt_strat]
        n_trials = 20
        refit_every = 5
        generator = OptimizeAcqfGenerator

        [LevelSetEstimation]
        beta = 3.98

        [GPClassificationModel]
        inducing_size = 10
        mean_covar_factory = default_mean_covar_factory

        [OptimizeAcqfGenerator]
        restarts = 10
        samps = 1000
        """
        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)

        self.assertTrue(isinstance(strat.strat_list[0].generator, SobolGenerator))
        self.assertTrue(isinstance(strat.strat_list[1].model, GPClassificationModel))
        self.assertTrue(strat.strat_list[1].generator.acqf is LevelSetEstimation)
        # since ProbitObjective() is turned into an obj, we check for keys and then vals
        self.assertTrue(
            set(strat.strat_list[1].generator.acqf_kwargs.keys())
            == {"beta", "target", "objective"}
        )
        self.assertTrue(strat.strat_list[1].generator.acqf_kwargs["target"] == 0.75)
        self.assertTrue(strat.strat_list[1].generator.acqf_kwargs["beta"] == 3.98)
        self.assertEqual(strat.strat_list[1].generator.acqf_kwargs["objective"], None)

        self.assertTrue(strat.strat_list[1].generator.restarts == 10)
        self.assertTrue(strat.strat_list[1].generator.samps == 1000)
        self.assertTrue(strat.strat_list[0].n_trials == 10)
        self.assertTrue(strat.strat_list[0].outcome_type == "single_probit")
        self.assertTrue(strat.strat_list[1].n_trials == 20)
        self.assertTrue(torch.all(strat.strat_list[0].lb == strat.strat_list[1].lb))
        self.assertTrue(torch.all(strat.strat_list[1].model.lb == torch.Tensor([0, 0])))
        self.assertTrue(torch.all(strat.strat_list[0].ub == strat.strat_list[1].ub))
        self.assertTrue(torch.all(strat.strat_list[1].model.ub == torch.Tensor([1, 1])))

    def test_1d_classification(self):
        """
        Just see if we memorize the training set
        """
        np.random.seed(1)
        torch.manual_seed(1)
        X, y = make_classification(
            n_features=1,
            n_redundant=0,
            n_informative=1,
            random_state=1,
            n_clusters_per_class=1,
        )
        X, y = torch.Tensor(X), torch.Tensor(y)

        model = GPClassificationModel(torch.Tensor([-3]), torch.Tensor([3]))
        model.fit(X, y)
        pred = (torch.sigmoid(model.posterior(X).mean) > 0.5).numpy()
        npt.assert_allclose(pred[:, 0], y)

    def test_1d_query(self):
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

        # Test sine function with period 4
        def test_fun(x):
            return np.sin(np.pi * x / 4)

        strat_list = [
            Strategy(
                lb=lb,
                ub=ub,
                n_trials=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    qUpperConfidenceBound,
                    acqf_kwargs={"beta": 1.96, "objective": GenericMCObjective(obj)},
                ),
                n_trials=n_opt,
            ),
        ]

        strat = SequentialStrategy(strat_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(norm.cdf(test_fun(next_x)))])

        # We expect the global max to be at (2, 1), the min at (-2, -1)
        fmax, argmax = strat.get_max()
        self.assertTrue(np.abs(fmax - 1) < 0.5)
        self.assertTrue(np.abs(argmax[0] - 2) < 0.5)

        fmin, argmin = strat.get_min()
        self.assertTrue(np.abs(fmin + 1) < 0.5)
        self.assertTrue(np.abs(argmin[0] + 2) < 0.5)

        # Query at x=2 should be f=1
        self.assertTrue(np.abs(strat.predict(torch.tensor([2]))[0] - 1) < 0.5)

        # Inverse query at val 1 should return (1,[2])
        val, loc = strat.inv_query(1.0, constraints={})
        self.assertTrue(np.abs(val - 1) < 0.5)
        self.assertTrue(np.abs(loc[0] - 2) < 0.5)


if __name__ == "__main__":
    unittest.main()
