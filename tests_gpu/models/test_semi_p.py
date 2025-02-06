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
        self.X = torch.Tensor(X).cuda()
        self.inducing_size = 10

    def test_analytic_lookahead_generation(self):
        floor = 0
        objective = FloorProbitObjective
        model = _semip_model_constructor(
            stim_dim=self.stim_dim,
            floor=floor,
            objective=objective,
        )
        model.cuda()

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
                stim_dim=self.stim_dim,
                floor=floor,
                objective=objective,
            )
            model.cuda()

            model.fit(train_x=self.X[:50], train_y=y[:50])

            pm, _ = model.predict(self.X[:50])
            pred = (link(pm) > 0.5).cpu().numpy()
            npt.assert_allclose(pred, y[:50].numpy(), atol=1)  # mismatch at most one

            model.update(self.X, y)

            pm, _ = model.predict(self.X[50:])
            pred = (link(pm) > 0.5).cpu().numpy()
            npt.assert_allclose(pred, y[50:].numpy(), atol=1)


if __name__ == "__main__":
    unittest.main()
