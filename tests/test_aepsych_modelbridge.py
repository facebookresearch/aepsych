#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import numpy as np
import torch
from aepsych.acquisition.lse import LevelSetEstimation
from aepsych.acquisition.mc_posterior_variance import (
    MCPosteriorVariance,
    MonotonicMCPosteriorVariance,
)
from aepsych.acquisition.monotonic_rejection import MonotonicMCLSE
from aepsych.modelbridge.monotonic import MonotonicSingleProbitModelbridge
from aepsych.modelbridge.single_probit import SingleProbitModelbridge
from aepsych.models.monotonic_rejection_gp import MonotonicRejectionGP
from aepsych.server import AEPsychServer
from aepsych.strategy import (
    SequentialStrategy,
    SobolStrategy,
    ModelWrapperStrategy,
)
from aepsych.utils import get_lse_contour
from botorch.acquisition import qUpperConfidenceBound
from botorch.acquisition.objective import GenericMCObjective
from scipy.stats import bernoulli, norm, pearsonr
from torch.distributions import Normal

from .common import f_1d, f_2d, cdf_new_novel_det, cdf_new_novel_det_3D


class SingleProbitModelbridgeModelBridgeTest(unittest.TestCase):
    def test_1d_single_probit_new_interface(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        lb = -4.0
        ub = 4.0

        model_list = [
            SobolStrategy(lb=lb, ub=ub, seed=seed, n_trials=n_init),
            ModelWrapperStrategy(
                modelbridge=SingleProbitModelbridge(lb=lb, ub=ub, dim=1),
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
            SobolStrategy(lb=lb, ub=ub, seed=seed, n_trials=n_init),
            ModelWrapperStrategy(
                modelbridge=SingleProbitModelbridge(lb=lb, ub=ub, dim=1),
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

    def test_1d_monotonic_single_probit_batched(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 2
        lb = -4.0
        ub = 4.0
        extra_acqf_args = {"target": 0.75, "beta": 3.98}
        model_list = [
            SobolStrategy(lb=lb, ub=ub, seed=seed, n_trials=n_init),
            ModelWrapperStrategy(
                modelbridge=MonotonicSingleProbitModelbridge(
                    lb=lb,
                    ub=ub,
                    dim=1,
                    extra_acqf_args=extra_acqf_args,
                    acqf=MonotonicMCLSE,
                ),
                n_trials=n_opt,
            ),
        ]

        strat = SequentialStrategy(model_list)
        while not strat.finished:
            next_x = strat.gen(num_points=2)
            strat.add_data(next_x, bernoulli.rvs(norm.cdf(next_x)).squeeze())

        self.assertEqual(strat.y.shape[0], n_init + n_opt)

        x = torch.linspace(-4, 4, 100)

        zhat, _ = strat.predict(x[:, None])

        # f(x) = x so we're just looking at corr between cdf(zhat) and cdf(x)
        self.assertTrue(
            pearsonr(norm.cdf(zhat.detach().numpy()).flatten(), norm.cdf(x).flatten())[
                0
            ]
            > 0.95
        )

    def test_1d_single_probit(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        lb = -4.0
        ub = 4.0

        model_list = [
            SobolStrategy(lb=lb, ub=ub, seed=seed, n_trials=n_init),
            ModelWrapperStrategy(
                modelbridge=SingleProbitModelbridge(lb=lb, ub=ub, dim=1),
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
            SobolStrategy(lb=lb, ub=ub, n_trials=n_init, seed=seed),
            ModelWrapperStrategy(
                modelbridge=SingleProbitModelbridge(
                    lb=lb, ub=ub, acqf=MCPosteriorVariance
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

    def test_1d_monotonic_single_probit_pure_exploration(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        lb = -4.0
        ub = 4.0

        strat_list = [
            SobolStrategy(lb=lb, ub=ub, n_trials=n_init, seed=seed),
            ModelWrapperStrategy(
                modelbridge=MonotonicSingleProbitModelbridge(
                    lb=lb,
                    ub=ub,
                    acqf=MonotonicMCPosteriorVariance,
                    model=MonotonicRejectionGP(
                        likelihood="probit-bernoulli", monotonic_idxs=[0]
                    ),
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
            SobolStrategy(lb=lb, ub=ub, dim=2, n_trials=n_init, seed=seed),
            ModelWrapperStrategy(
                modelbridge=SingleProbitModelbridge(
                    lb=lb,
                    ub=ub,
                    dim=2,
                    acqf=MCPosteriorVariance,
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

        mc_objective = GenericMCObjective(obj)
        extra_acqf_args = {"objective": mc_objective, "beta": 1.96}
        strat_list = [
            SobolStrategy(lb=lb, ub=ub, n_trials=n_init, seed=seed),
            ModelWrapperStrategy(
                modelbridge=SingleProbitModelbridge(
                    lb=lb,
                    ub=ub,
                    acqf=qUpperConfidenceBound,
                    extra_acqf_args=extra_acqf_args,
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

        mc_objective = GenericMCObjective(obj)
        extra_acqf_args = {"objective": mc_objective, "beta": 1.96}
        strat_list = [
            SobolStrategy(lb=lb, ub=ub, n_trials=n_init, seed=seed),
            ModelWrapperStrategy(
                modelbridge=SingleProbitModelbridge(
                    lb=lb,
                    ub=ub,
                    acqf=qUpperConfidenceBound,
                    extra_acqf_args=extra_acqf_args,
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
            SobolStrategy(lb=lb, ub=ub, n_trials=n_init, seed=seed),
            ModelWrapperStrategy(
                modelbridge=SingleProbitModelbridge(
                    lb=lb,
                    ub=ub,
                    acqf=LevelSetEstimation,
                    extra_acqf_args=extra_acqf_args,
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

    def test_monotonic_jnd_single(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 150
        n_opt = 1
        lb = -4.0
        ub = 4.0

        # target is in z space not phi(z) space, maybe that's
        # weird
        extra_acqf_args = {"target": 0.5, "beta": 3.98}

        strat_list = [
            SobolStrategy(lb=lb, ub=ub, n_trials=n_init, seed=seed),
            ModelWrapperStrategy(
                modelbridge=MonotonicSingleProbitModelbridge(
                    lb=lb,
                    ub=ub,
                    acqf=MonotonicMCPosteriorVariance,
                    extra_acqf_args=extra_acqf_args,
                    monotonic_idxs=[0],
                ),
                n_trials=n_opt,
            ),
        ]

        strat = SequentialStrategy(strat_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(norm.cdf(next_x / 1.5))])

        x = torch.linspace(-4, 4, 101)
        zhat, _ = strat.predict(x[:, None])

        # we eval the jnd in the middle of the space (where the true)
        # threshold is since we're aiming for 0.5 threshold.
        # we expect jnd close to the target to be close to the correct
        # jnd (1.5)
        # TODO mock this instead to make it cleaner, faster,
        # and more reliable
        jnd_step = strat.get_jnd(grid=x[:, None], method="step")
        est_jnd_step = jnd_step[50]
        # looser test because step-jnd is hurt more by reverting to the mean
        self.assertTrue(np.abs(est_jnd_step - 1.5) < 0.75)

        jnd_taylor = strat.get_jnd(grid=x[:, None], method="taylor")
        est_jnd_taylor = jnd_taylor[50]
        self.assertTrue(np.abs(est_jnd_taylor - 1.5) < 0.25)

    def test_2d_novel_det(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        lb = [-1, -1]
        ub = [1, 1]

        extra_acqf_args = {"target": 0.75, "beta": 1.96}

        strat_list = [
            SobolStrategy(lb=lb, ub=ub, dim=2, n_trials=n_init, seed=seed),
            ModelWrapperStrategy(
                modelbridge=MonotonicSingleProbitModelbridge(
                    lb=lb,
                    ub=ub,
                    dim=2,
                    acqf=MonotonicMCLSE,
                    extra_acqf_args=extra_acqf_args,
                    monotonic_idxs=[1],
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
        x1 = torch.linspace(-1, 1, 30)
        x2_hat = get_lse_contour(phi_post_mean, x1, level=0.75, lb=-1.0, ub=1.0)

        true_f = cdf_new_novel_det(xy)
        true_x2 = x1[np.argmin(((true_f - 0.75) ** 2).reshape(30, 30), axis=1)].numpy()
        self.assertTrue(np.mean(np.abs(x2_hat - true_x2)) < 0.15)

    def test_3d_novel_det(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 100
        n_opt = 1
        lb = [-1, -1, -1]
        ub = [1, 1, 1]

        extra_acqf_args = {"target": 0.75, "beta": 1.96}

        strat_list = [
            SobolStrategy(lb=lb, ub=ub, dim=3, n_trials=n_init, seed=seed),
            ModelWrapperStrategy(
                modelbridge=MonotonicSingleProbitModelbridge(
                    lb=lb,
                    ub=ub,
                    dim=3,
                    acqf=MonotonicMCLSE,
                    extra_acqf_args=extra_acqf_args,
                    monotonic_idxs=[2],
                ),
                n_trials=n_opt,
            ),
        ]

        strat = SequentialStrategy(strat_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(cdf_new_novel_det_3D(next_x))])

        gridsize_freq = 17
        gridsize_chan = 13
        gridsize_amp = 15
        extent = np.c_[strat.lb, strat.ub].reshape(-1)
        x = torch.Tensor(
            np.mgrid[
                extent[0] : extent[1] : gridsize_freq * 1j,
                extent[2] : extent[3] : gridsize_chan * 1j,
                extent[4] : extent[5] : gridsize_amp * 1j,
            ]
            .reshape(3, -1)
            .T
        )

        post_mean, _ = strat.predict(torch.Tensor(x))
        phi_post_mean = norm.cdf(
            post_mean.reshape(gridsize_freq, gridsize_chan, gridsize_amp)
            .detach()
            .numpy()
        )
        mono_grid = torch.linspace(-1, 1, gridsize_amp)

        x2_hat = get_lse_contour(phi_post_mean, mono_grid, level=0.75, lb=-1.0, ub=1.0)
        true_f = cdf_new_novel_det_3D(x).reshape(
            gridsize_freq, gridsize_chan, gridsize_amp
        )
        true_x2 = get_lse_contour(true_f, mono_grid, level=0.75, lb=-1.0, ub=1.0)

        self.assertTrue(np.mean(np.abs(x2_hat - true_x2)) < 0.15)

    def test_2d_single_probit(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 150
        n_opt = 1
        lb = np.r_[-1, -1]
        ub = np.r_[1, 1]

        model_list = [
            SobolStrategy(lb=lb, ub=ub, seed=seed, n_trials=n_init, dim=2),
            ModelWrapperStrategy(
                modelbridge=SingleProbitModelbridge(lb=lb, ub=ub, dim=2),
                n_trials=n_opt,
            ),
        ]

        strat = SequentialStrategy(model_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(f_2d(next_x[None, :]))])

        xy = np.mgrid[-1:1:30j, -1:1:30j].reshape(2, -1).T
        zhat, _ = strat.predict(torch.Tensor(xy))

        self.assertTrue(np.all(np.abs(xy[np.argmax(zhat.detach().numpy())]) < 0.5))

    @unittest.skip("SLOW")
    def test_2d_single_probit_stopping_criterion(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 100
        n_opt = 10
        lb = [-1, -1]
        ub = [1, 1]

        extra_acqf_args = {"target": 0.75, "beta": 1.96, "lse_threshold": 0.2}
        strat_list = [
            SobolStrategy(lb=lb, ub=ub, dim=2, n_trials=n_init, seed=seed),
            ModelWrapperStrategy(
                modelbridge=MonotonicSingleProbitModelbridge(
                    lb=lb,
                    ub=ub,
                    dim=2,
                    acqf=MonotonicMCLSE,
                    extra_acqf_args=extra_acqf_args,
                    monotonic_idxs=[1],
                ),
                n_trials=n_opt,
            ),
        ]

        strat = SequentialStrategy(strat_list)

        while not strat.finished:
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(cdf_new_novel_det(next_x))])

        self.assertTrue(strat.y.shape[0] < 25)

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
            SobolStrategy(lb=lb, ub=ub, seed=seed, n_trials=n_init),
            ModelWrapperStrategy(
                modelbridge=SingleProbitModelbridge(lb=lb, ub=ub),
                n_trials=n_opt,
            ),
        ]

        strat = SequentialStrategy(model_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(norm.cdf(f_1d(next_x)))])

        with self.assertWarns(RuntimeWarning):
            strat.gen()


class SingleProbitModelbridgeServerTest(unittest.TestCase):
    def test_1d_single_server(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1

        experiment_config = {
            "common": {"lb": [-4.0], "ub": [4.0]},
            "experiment": {"parnames": "[x]"},
            "SobolStrategy": {"n_trials": n_init},
            "ModelWrapperStrategy": {"n_trials": n_opt},
        }

        server = AEPsychServer()
        server.configure(config_dict=experiment_config)

        for _i in range(n_init + n_opt):
            next_config = server.ask()
            next_y = bernoulli.rvs(f_1d(np.c_[next_config["x"]]))
            server.tell(config=next_config, outcome=next_y)

        x = torch.linspace(-4, 4, 100)
        zhat, _ = server.strat.predict(x)
        self.assertTrue(np.abs(x[np.argmax(zhat.detach().numpy())]) < 0.5)

    def test_server_multistrat(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)

        config1 = {
            "common": {"lb": [-4.0], "ub": [4.0]},
            "experiment": {"parnames": "[x]"},
            "SobolStrategy": {"n_trials": 15},
            "ModelWrapperStrategy": {"n_trials": 5},
        }

        config2 = {
            "common": {"lb": [-2.0], "ub": [20.0]},
            "experiment": {"parnames": "[x]"},
            "SobolStrategy": {"n_trials": 30},
            "ModelWrapperStrategy": {"n_trials": 5},
        }

        config1_msg = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_dict": config1},
        }

        config2_msg = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_dict": config2},
        }

        resume_msg = {"type": "resume", "version": "0.01", "message": {"strat_id": 0}}
        server = AEPsychServer()

        resp0 = server.versioned_handler(config1_msg)

        self.assertTrue(resp0 == 0)
        strat1 = server.strat
        resp1 = server.versioned_handler(config2_msg)
        self.assertTrue(resp1 == 1)
        self.assertTrue(server.strat != strat1)
        resp2 = server.versioned_handler(resume_msg)
        self.assertTrue(resp2 == 0)
        self.assertTrue(server.strat == strat1)

    def test_config_to_tensor(self):
        n_init = 1
        n_opt = 1
        experiment_config = {
            "common": {"lb": [-1.0], "ub": [1.0]},
            "experiment": {"parnames": "[x]"},
            "SobolStrategy": {"n_trials": n_init},
            "ModelWrapperStrategy": {"n_trials": n_opt},
        }

        server = AEPsychServer()
        server.configure(config_dict=experiment_config)

        conf = server.ask()

        self.assertTrue(server._config_to_tensor(conf).shape == (1, 1))

        experiment_config = {
            "common": {"lb": [-1.0, -1], "ub": [1.0, 1]},
            "experiment": {"parnames": "[x, y]"},
            "SobolStrategy": {"n_trials": n_init},
            "ModelWrapperStrategy": {"n_trials": n_opt},
        }

        server = AEPsychServer()
        server.configure(config_dict=experiment_config)

        conf = server.ask()

        self.assertTrue(server._config_to_tensor(conf).shape == (1, 2))

        experiment_config = {
            "common": {"lb": [-1.0, -1.0, -1.0], "ub": [1.0, 1.0, 1.0]},
            "experiment": {"parnames": "[x, y, z]"},
            "SobolStrategy": {"n_trials": n_init},
            "ModelWrapperStrategy": {"n_trials": n_opt},
        }

        server = AEPsychServer()
        server.configure(config_dict=experiment_config)
        conf = server.ask()

        self.assertTrue(server._config_to_tensor(conf).shape == (1, 3))

    def test_serialization_1d(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 3
        n_opt = 1
        experiment_config = {
            "common": {"lb": [-4.0], "ub": [4.0]},
            "experiment": {"parnames": "[x]"},
            "SobolStrategy": {"n_trials": n_init},
            "ModelWrapperStrategy": {"n_trials": n_opt},
        }

        server = AEPsychServer()
        server.configure(config_dict=experiment_config)
        for _i in range(n_init + n_opt):
            next_config = server.ask()
            next_y = bernoulli.rvs(norm.cdf(f_1d(np.c_[next_config["x"]])))
            server.tell(config=next_config, outcome=next_y)

        import dill

        # just make sure it works
        try:
            s = dill.dumps(server)
            server2 = dill.loads(s)
            next_config = server2.ask()
            next_y = bernoulli.rvs(norm.cdf(f_1d(np.c_[next_config["x"]])))
            server2.tell(config=next_config, outcome=next_y)

        except Exception:
            self.fail()

    def test_serialization_2d(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 3
        n_opt = 1

        experiment_config = {
            "common": {"lb": [-4.0, -4.0], "ub": [4.0, 4.0]},
            "experiment": {"parnames": "[x, y]"},
            "SobolStrategy": {"n_trials": n_init},
            "ModelWrapperStrategy": {"n_trials": n_opt},
        }

        server = AEPsychServer()
        server.configure(config_dict=experiment_config)
        for _i in range(n_init + n_opt):
            next_config = server.ask()
            next_y = bernoulli.rvs(norm.cdf(next_config["x"][0] + next_config["y"][0]))
            server.tell(config=next_config, outcome=next_y)

        import dill

        # just make sure it works
        try:
            s = dill.dumps(server)
            server2 = dill.loads(s)
            next_config = server2.ask()
            next_y = bernoulli.rvs(norm.cdf(f_1d(np.c_[next_config["x"]])))
            server2.tell(config=next_config, outcome=next_y)

        except Exception:
            self.fail()
