#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import numpy.testing as npt
import torch
from aepsych.acquisition import MCPosteriorVariance
from aepsych.acquisition.lookahead import GlobalMI
from aepsych.acquisition.objective import (
    FloorGumbelObjective,
    FloorLogitObjective,
    FloorProbitObjective,
    ProbitObjective,
)
from aepsych.acquisition.objective.semi_p import (
    SemiPProbabilityObjective,
    SemiPThresholdObjective,
)
from aepsych.generators import OptimizeAcqfGenerator, SobolGenerator
from aepsych.likelihoods import BernoulliObjectiveLikelihood
from aepsych.likelihoods.semi_p import LinearBernoulliLikelihood
from aepsych.models import HadamardSemiPModel, SemiParametricGPModel
from aepsych.models.semi_p import _hadamard_mvn_approx, semi_p_posterior_transform
from aepsych.strategy import SequentialStrategy, Strategy
from aepsych.utils import make_scaled_sobol
from gpytorch.distributions import MultivariateNormal
from parameterized import parameterized


def _hadamard_model_constructor(
    stim_dim,
    floor,
    objective=FloorLogitObjective,
):
    return HadamardSemiPModel(
        dim=2,
        stim_dim=stim_dim,
        likelihood=BernoulliObjectiveLikelihood(objective=objective(floor=floor)),
        inducing_size=10,
        max_fit_time=0.5,
    )


def _semip_model_constructor(
    stim_dim,
    floor,
    objective=FloorLogitObjective,
):
    return SemiParametricGPModel(
        dim=2,
        stim_dim=stim_dim,
        likelihood=LinearBernoulliLikelihood(objective=objective(floor=floor)),
        inducing_size=10,
    )


links = [FloorLogitObjective, FloorProbitObjective, FloorGumbelObjective]
floors = [0, 0.3, 0.5]
constructors = [_semip_model_constructor, _hadamard_model_constructor]
test_configs = [[FloorLogitObjective, 0.3, _hadamard_model_constructor]]
# test_configs = list(product(links, floors, constructors)) # TODO too slow


class SemiPSmokeTests(unittest.TestCase):
    def setUp(self):
        self.seed = 1
        self.stim_dim = 0
        self.context_dim = 1
        np.random.seed(1)
        torch.manual_seed(1)
        X = np.random.randn(100, 2) / 3
        xcontext = X[..., self.context_dim]
        xintensity = X[..., self.stim_dim]
        # polynomial context
        slope = xcontext - 0.7 * xcontext**2 + 0.3 * xcontext**3 - 0.1 * xcontext**4
        intercept = (
            xcontext + 0.03 * xcontext**5 - 0.2 * xcontext**3 - 0.7 * xcontext**4
        )
        # multiply by intensity
        self.f = torch.Tensor(slope * (intercept + xintensity)).unsqueeze(-1)
        X[:, 0] = X[:, 0] * 100
        X[:, 1] = X[:, 1] / 100
        self.lb = torch.tensor([-100.0, -0.01])
        self.ub = torch.tensor([100.0, 0.01])
        self.X = torch.Tensor(X)
        self.inducing_size = 10

    @parameterized.expand(
        [(SemiPThresholdObjective(target=0.75),), (SemiPProbabilityObjective(),)]
    )
    def test_mc_generation(self, objective):
        # no objective here, the objective for `gen` is not the same as the objective
        # for the likelihood in this case
        model = SemiParametricGPModel(
            dim=2,
            stim_dim=self.stim_dim,
            likelihood=LinearBernoulliLikelihood(),
            inducing_size=10,
        )

        generator = OptimizeAcqfGenerator(
            acqf=MCPosteriorVariance,
            acqf_kwargs={"objective": objective},
            max_gen_time=0.1,
            lb=self.lb,
            ub=self.ub,
        )

        y = torch.bernoulli(model.likelihood.objective(self.f))
        model.set_train_data(
            self.X[:10], y[:10]
        )  # no need to fit for checking gen shapes
        next_x = generator.gen(num_points=1, model=model)
        self.assertEqual(
            next_x.shape,
            (
                1,
                2,
            ),
        )

    def test_analytic_lookahead_generation(self):
        floor = 0
        objective = FloorProbitObjective
        model = _semip_model_constructor(
            stim_dim=self.stim_dim,
            floor=floor,
            objective=objective,
        )

        generator = OptimizeAcqfGenerator(
            acqf=GlobalMI,
            acqf_kwargs={
                "posterior_transform": semi_p_posterior_transform,
                "target": 0.75,
                "query_set_size": 100,
                "Xq": make_scaled_sobol(self.lb, self.ub, 100),
                "lb": self.lb,
                "ub": self.ub,
            },
            max_gen_time=0.2,
            lb=self.lb,
            ub=self.ub,
        )
        link = objective(floor=floor)
        y = torch.bernoulli(link(self.f))

        model.set_train_data(
            self.X[:10], y[:10]
        )  # no need to fit for checking gen shapes

        next_x = generator.gen(num_points=1, model=model)
        self.assertEqual(
            next_x.shape,
            (
                1,
                2,
            ),
        )

    @parameterized.expand(test_configs)
    @unittest.skip("Slow smoke test, TODO speed me up")
    def test_memorize_data(self, objective, floor, model_constructor):
        """
        see approximate accuracy on easy logistic ps that only varies in 1d
        (no slope and intercept)
        accuracy determined by average performance on training data
        """
        with self.subTest(
            objective=objective.__name__,
            floor=floor,
            model_constructor=model_constructor,
        ):
            link = objective(floor=floor)
            y = torch.bernoulli(link(self.f))

            model = model_constructor(
                inducing_point_method=self.inducing_point_method,
                stim_dim=self.stim_dim,
                floor=floor,
                objective=objective,
            )

            model.fit(train_x=self.X[:50], train_y=y[:50])

            pm, _ = model.predict(self.X[:50])
            pred = (link(pm) > 0.5).numpy()
            npt.assert_allclose(pred, y[:50].numpy(), atol=1)  # mismatch at most one

            model.update(self.X, y)

            pm, _ = model.predict(self.X[50:])
            pred = (link(pm) > 0.5).numpy()
            npt.assert_allclose(pred, y[50:].numpy(), atol=1)

    @parameterized.expand([(_semip_model_constructor,), (_hadamard_model_constructor,)])
    def test_prediction_shapes(self, model_constructor):
        n_opt = 1
        lb = torch.tensor([-1.0, -1.0])
        ub = torch.tensor([1.0, 1.0])

        with self.subTest(model_constructor=model_constructor):
            model = model_constructor(
                stim_dim=self.stim_dim,
                floor=0,
            )

            strat_list = [
                Strategy(
                    lb=lb,
                    ub=ub,
                    model=model,
                    generator=SobolGenerator(lb=lb, ub=ub, seed=self.seed),
                    min_asks=n_opt,
                    stimuli_per_trial=1,
                    outcome_types=["binary"],
                ),
            ]
            strat = SequentialStrategy(strat_list)

            train_x = torch.tensor([[0.0, 0.0], [2.0, 1.0], [2.0, 2.0]])
            train_y = torch.tensor([1.0, 1.0, 0.0])
            model.fit(train_x=train_x, train_y=train_y)
            f, var = model.predict(train_x)
            self.assertEqual(f.shape, torch.Size([3]))
            self.assertEqual(var.shape, torch.Size([3]))

            p, pvar = model.predict(train_x, probability_space=True)
            self.assertEqual(p.shape, torch.Size([3]))
            self.assertEqual(pvar.shape, torch.Size([3]))

            if isinstance(model, SemiParametricGPModel):
                samps = model.sample(train_x, 11, probability_space=True)
                self.assertEqual(samps.shape, torch.Size([11, 3]))
                post = model.posterior(train_x)
                self.assertEqual(post.mvn.mean.shape, torch.Size([2, 3]))
                self.assertTrue(torch.equal(post.Xi, torch.tensor([0.0, 2.0, 2.0])))
                samps = post.rsample(sample_shape=torch.Size([6]))
                # samps should be n_samp x 2 (slope, intercept) * 3 (xshape)
                self.assertEqual(samps.shape, torch.Size([6, 2, 3]))

                # now check posterior samp sizes. They have
                # an extra dim (since it's 1d outcome), which
                # model.sample squeezes, except for thresh sampling
                # which is already squeezed by the threshold objective
                # TODO be more consistent with how we use dims
                post = model.posterior(train_x)
                p_samps = post.sample_p(torch.Size([6]))
                self.assertEqual(p_samps.shape, torch.Size([6, 1, 3]))
                f_samps = post.sample_f(torch.Size([6]))
                self.assertEqual(f_samps.shape, torch.Size([6, 1, 3]))
                thresh_samps = post.sample_thresholds(
                    threshold_level=0.75, sample_shape=torch.Size([6])
                )
                self.assertEqual(thresh_samps.shape, torch.Size([6, 3]))

            strat.add_data(train_x, train_y)
            Xopt = strat.gen()
            self.assertEqual(Xopt.shape, torch.Size([1, 2]))

    @parameterized.expand([(_semip_model_constructor,), (_hadamard_model_constructor,)])
    def test_reset_variational_strategy(self, model_constructor):
        stim_dim = 0
        with self.subTest(model_constructor=model_constructor):
            model = model_constructor(
                stim_dim=stim_dim,
                floor=0,
            )
            link = FloorLogitObjective(floor=0)
            y = torch.bernoulli(link(self.f))

            variational_params_before = [
                v.clone().detach().numpy() for v in model.variational_parameters()
            ]
            induc_before = model.variational_strategy.inducing_points

            model.fit(torch.Tensor(self.X[:15]), torch.Tensor(y[:15]))

            variational_params_after = [
                v.clone().detach().numpy() for v in model.variational_parameters()
            ]
            induc_after = model.variational_strategy.inducing_points

            model._reset_variational_strategy()

            variational_params_reset = [
                v.clone().detach().numpy() for v in model.variational_parameters()
            ]

            # before should be different from after
            if induc_before.shape == induc_after.shape:  # Not same can't fail
                self.assertFalse(np.allclose(induc_before, induc_after))

            for before, after in zip(
                variational_params_before, variational_params_after
            ):
                if before.shape == after.shape:  # Not same can't fail
                    self.assertFalse(np.allclose(before, after))

            for after, reset in zip(variational_params_after, variational_params_reset):
                self.assertFalse(np.allclose(after, reset))

    def test_slope_mean_setting(self):
        for slope_mean in (2, 4):
            model = SemiParametricGPModel(
                dim=2,
                stim_dim=self.stim_dim,
                likelihood=LinearBernoulliLikelihood(),
                inducing_size=self.inducing_size,
                slope_mean=slope_mean,
            )
            with self.subTest(model=model, slope_mean=slope_mean):
                self.assertEqual(model.mean_module.constant[-1], slope_mean)
            model = HadamardSemiPModel(
                dim=2,
                stim_dim=self.stim_dim,
                likelihood=BernoulliObjectiveLikelihood(objective=ProbitObjective()),
                inducing_size=self.inducing_size,
                slope_mean=slope_mean,
            )
            with self.subTest(model=model, slope_mean=slope_mean):
                self.assertEqual(model.slope_mean_module.constant.item(), slope_mean)


class HadamardSemiPtest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        torch.manual_seed(1)
        stim_dim = 0
        X = torch.randn(100, 2)
        self.X = X
        link = ProbitObjective()
        self.y = torch.bernoulli(link(X[:, stim_dim]))

    def test_reset_hyperparams(self):
        model = HadamardSemiPModel(
            dim=2,
            inducing_size=20,
        )

        slope_os_before = model.slope_covar_module.outputscale.clone().detach().numpy()
        offset_os_before = (
            model.offset_covar_module.outputscale.clone().detach().numpy()
        )
        slope_ls_before = (
            model.slope_covar_module.base_kernel.lengthscale.clone().detach().numpy()
        )
        offset_ls_before = (
            model.offset_covar_module.base_kernel.lengthscale.clone().detach().numpy()
        )

        model.fit(self.X[:15], self.y[:15])

        slope_os_after = model.slope_covar_module.outputscale.clone().detach().numpy()
        offset_os_after = model.offset_covar_module.outputscale.clone().detach().numpy()
        slope_ls_after = (
            model.slope_covar_module.base_kernel.lengthscale.clone().detach().numpy()
        )
        offset_ls_after = (
            model.offset_covar_module.base_kernel.lengthscale.clone().detach().numpy()
        )

        model._reset_hyperparameters()

        slope_os_reset = model.slope_covar_module.outputscale.clone().detach().numpy()
        offset_os_reset = model.offset_covar_module.outputscale.clone().detach().numpy()
        slope_ls_reset = (
            model.slope_covar_module.base_kernel.lengthscale.clone().detach().numpy()
        )
        offset_ls_reset = (
            model.offset_covar_module.base_kernel.lengthscale.clone().detach().numpy()
        )

        # before should be different from after and after should be different
        # from reset but before and reset should be same
        self.assertFalse(np.allclose(slope_os_before, slope_os_after))
        self.assertFalse(np.allclose(slope_os_after, slope_os_reset))
        self.assertTrue(np.allclose(slope_os_before, slope_os_reset))
        self.assertFalse(np.allclose(slope_ls_before, slope_ls_after))
        self.assertFalse(np.allclose(slope_ls_after, slope_ls_reset))
        self.assertTrue(np.allclose(slope_ls_before, slope_ls_reset))

        self.assertFalse(np.allclose(offset_os_before, offset_os_after))
        self.assertFalse(np.allclose(offset_os_after, offset_os_reset))
        self.assertTrue(np.allclose(offset_os_before, offset_os_reset))
        self.assertFalse(np.allclose(offset_ls_before, offset_ls_after))
        self.assertFalse(np.allclose(offset_ls_after, offset_ls_reset))
        self.assertTrue(np.allclose(offset_ls_before, offset_ls_reset))

    def _make_psd_matrix(self, size):
        L = torch.randn((size, size))
        return L @ L.T

    def test_normal_approx(self):
        np.random.seed(123)
        torch.manual_seed(123)

        npoints = 10

        def make_samp_and_approx_mvns(kcov_scale=1.0, ccov_scale=1.0):
            X = torch.randn(npoints)
            kmean = torch.randn(npoints)
            cmean = torch.randn(npoints)
            kcov = self._make_psd_matrix(npoints) * kcov_scale
            ccov = self._make_psd_matrix(npoints) * ccov_scale

            k_mvn = MultivariateNormal(kmean, kcov)
            c_mvn = MultivariateNormal(cmean, ccov)

            ksamps = k_mvn.sample(torch.Size([1000]))
            csamps = c_mvn.sample(torch.Size([1000]))

            samp_mean = (ksamps * (X + csamps)).mean(0)
            samp_cov = (ksamps * (X + csamps)).T.cov()

            approx_mean, approx_cov = _hadamard_mvn_approx(
                X, slope_mean=kmean, slope_cov=kcov, offset_mean=cmean, offset_cov=ccov
            )
            return samp_mean, samp_cov, approx_mean, approx_cov

        # verify that as kvar approaches 0, approx improves on average
        mean_errs = []
        cov_errs = []
        for kcov_scale in [1e-5, 1e-2, 1]:
            mean_err = 0
            cov_err = 0
            for _rep in range(100):
                (
                    samp_mean,
                    samp_cov,
                    approx_mean,
                    approx_cov,
                ) = make_samp_and_approx_mvns(kcov_scale=kcov_scale)
                mean_err += (samp_mean - approx_mean).abs().mean().item()
                cov_err += (samp_cov - approx_cov).abs().mean().item()
            mean_errs.append(mean_err / 100)
            cov_errs.append(cov_err / 100)
        npt.assert_equal(mean_errs, sorted(mean_errs))
        npt.assert_equal(cov_errs, sorted(cov_errs))

        # verify that as cvar approaches 0, approx improves on average
        mean_errs = []
        cov_errs = []
        for ccov_scale in [1e-5, 1e-2, 1]:
            mean_err = 0
            cov_err = 0
            for _rep in range(100):
                (
                    samp_mean,
                    samp_cov,
                    approx_mean,
                    approx_cov,
                ) = make_samp_and_approx_mvns(ccov_scale=ccov_scale)
                mean_err += (samp_mean - approx_mean).abs().mean().item()
                cov_err += (samp_cov - approx_cov).abs().mean().item()
            mean_errs.append(mean_err / 100)
            cov_errs.append(cov_err / 100)

        npt.assert_equal(mean_errs, sorted(mean_errs))
        npt.assert_equal(cov_errs, sorted(cov_errs))


if __name__ == "__main__":
    unittest.main()
