#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from aepsych import Config
from aepsych.models import GPClassificationModel, GPRegressionModel, IndependentGPsModel
from aepsych.models.transformed_posteriors import BernoulliProbitProbabilityPosterior
from aepsych.models.utils import dim_grid
from aepsych.strategy import SequentialStrategy
from botorch.posteriors import PosteriorList
from torch.nn import ModuleDict


def f_1d(x, mu=0):
    """
    latent is just a gaussian bump at mu
    """
    return np.exp(-((x - mu) ** 2))


def f_2d(x, target=None):
    """
    Distance to target
    """
    if target is None:
        target = torch.tensor([0.0, 0.0])

    if x.ndim > 1:
        return torch.exp(-torch.linalg.vector_norm(x - target, dim=1))

    return torch.exp(-torch.linalg.vector_norm(x - target))


class IndependentGPsModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1)
        np.random.seed(1)

        x = torch.linspace(-3, 3, 50)
        cls.X = x.unsqueeze(-1)
        cls.y = x
        cls.y_p = f_1d(x)
        cls.y_c = torch.bernoulli(cls.y_p)

        cls.m1 = GPRegressionModel(1)
        cls.m1.fit(cls.X, cls.y)

        cls.m2 = GPClassificationModel(1)
        cls.m2.fit(
            cls.X,
            cls.y_c,
        )

        md = ModuleDict({"m1": GPRegressionModel(1), "m2": GPClassificationModel(1)})
        cls.igpm = IndependentGPsModel(md)
        cls.igpm.fit(
            cls.X,
            torch.hstack((cls.y.unsqueeze(-1), cls.y_c.unsqueeze(-1))),
        )

    @staticmethod
    def assert_close(x1, x2, rtol=1e-4, atol=1e-4):
        return torch.testing.assert_close(x1, x2, rtol=rtol, atol=atol)

    def test_predict(self):
        pred1, var1 = self.m1.predict(self.X)
        pred2, var2 = self.m2.predict(self.X)
        predm, varm = self.igpm.predict(self.X)

        # Verify shapes of predict are correct
        self.assertEqual(predm.shape[0], len(self.X))
        self.assertEqual(varm.shape[0], len(self.X))

        self.assertEqual(predm.shape[1], 2)
        self.assertEqual(varm.shape[1], 2)

        # Verify predictions are close (they may not be exact because of stochasticity in model-fitting)
        self.assert_close(predm[:, 0], pred1)
        self.assert_close(predm[:, 1], pred2, rtol=1e-3, atol=1e-3)  # classif is rough
        self.assert_close(varm[:, 0], var1)
        self.assert_close(varm[:, 1], var2, rtol=1e-3, atol=1e-3)

    def test_predict_transform(self):
        pred1t, var1t = self.m1.predict_transform(self.X)
        pred2t, var2t = self.m2.predict_transform(self.X)
        predmt, varmt = self.igpm.predict_transform(self.X)

        # Verify shapes of predict are correct
        self.assertEqual(predmt.shape[0], len(self.X))
        self.assertEqual(varmt.shape[0], len(self.X))

        self.assertEqual(predmt.shape[1], 2)
        self.assertEqual(varmt.shape[1], 2)

        # Verify predictions are close (they may not be exact because of stochasticity in model-fitting)
        self.assert_close(predmt[:, 0], pred1t)
        self.assert_close(predmt[:, 1], pred2t)
        self.assert_close(varmt[:, 0], var1t)
        self.assert_close(varmt[:, 1], var2t)

    def test_predict_transform_non_default(self):
        predmt, varmt = self.igpm.predict_transform(
            self.X,
            transform_map={"m1": BernoulliProbitProbabilityPosterior, "m2": None},
        )

        self.assertEqual(predmt.shape[0], len(self.X))
        self.assertEqual(varmt.shape[0], len(self.X))

        self.assertEqual(predmt.shape[1], 2)
        self.assertEqual(varmt.shape[1], 2)

        self.assertTrue((predmt[:, 0] >= 0).all())
        self.assertTrue((predmt[:, 0] <= 1).all())

        self.assertTrue(predmt[:, 1].min() < 0)
        self.assertTrue(predmt[:, 1].max() > 0)

    def test_from_config(self):
        config_str = """
        [common]
        parnames = [foo, bar]
        outcome_names = [baz, qux]
        outcome_types = [binary, continuous]
        stimuli_per_trial = 1
        strategy_names = [init_strat]

        [init_strat]
        model = IndependentGPsModel

        [IndependentGPsModel]
        models = [model1, model2, GPClassificationModel]

        [model1]
        class = GPClassificationModel

        [model2]
        class = GPRegressionModel

        [foo]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1

        [bar]
        par_type = continuous
        lower_bound = 0
        upper_bound = 1
        """
        config = Config(config_str=config_str)
        m = IndependentGPsModel.from_config(config)

        self.assertIn("model1", m.models)
        self.assertIsInstance(m.models["model1"], GPClassificationModel)

        self.assertIn("model2", m.models)
        self.assertIsInstance(m.models["model2"], GPRegressionModel)

        self.assertIn("GPClassificationModel", m.models)
        self.assertIsInstance(m.models["GPClassificationModel"], GPClassificationModel)

    def test_num_outputs(self):
        self.assertEqual(
            self.igpm.num_outputs, self.m1.num_outputs + self.m2.num_outputs
        )

    def test_posterior(self):
        with torch.no_grad():
            postigpm = self.igpm.posterior(self.X)
            self.assertIsInstance(postigpm, PosteriorList)

            post1 = self.m1.posterior(self.X)
            post2 = self.m2.posterior(self.X)

            self.assert_close(post1.mean.squeeze(), postigpm.mean[..., 0].squeeze())
            self.assert_close(
                post2.mean.squeeze(),
                postigpm.mean[..., 1].squeeze(),
                atol=1e-3,
                rtol=1e-3,
            )  # GPClassification is rough

    def test_sample(self):
        with torch.no_grad():
            samples1 = self.m1.sample(self.X, 1000).mean(axis=0)
            samples1Again = self.m1.sample(self.X, 1000).mean(axis=0)
            samples2 = self.m2.sample(self.X, 1000).mean(axis=0)
            samples2Again = self.m2.sample(self.X, 1000).mean(axis=0)
            samplesigpm = self.igpm.sample(self.X, 1000).mean(axis=0)

            # Sampling is pretty noisy so tolerance is based on separate models reruns
            samples1Tolerance = torch.abs(samples1 - samples1Again).max()
            samples2Tolerance = torch.abs(samples2 - samples2Again).max()

            self.assert_close(
                samples1.mean(axis=0),
                samplesigpm[..., 0].mean(axis=0),
                rtol=samples1Tolerance,
                atol=samples1Tolerance,
            )
            self.assert_close(
                samples2.mean(axis=0),
                samplesigpm[..., 1].mean(axis=0),
                rtol=samples2Tolerance,
                atol=samples2Tolerance,
            )


class IndependentGPStratTest(unittest.TestCase):
    def test_end_to_end(self):
        torch.manual_seed(1)
        np.random.seed(1)

        config_str = """
            [common]
            parnames = [foo, bar]
            outcome_names = [baz, qux]
            outcome_types = [binary, continuous]
            stimuli_per_trial = 1
            strategy_names = [init_strat, opt_strat]

            [foo]
            par_type = continuous
            lower_bound = -2
            upper_bound = 2

            [bar]
            par_type = continuous
            lower_bound = -2
            upper_bound = 2

            [init_strat]
            generator = SobolGenerator
            min_asks = 150

            [opt_strat]
            generator = IndependentOptimizeAcqfGenerator
            model = IndependentGPsModel
            min_asks = 1

            [IndependentOptimizeAcqfGenerator]
            generators = [BazGen, QuxGen]

            [BazGen]
            class = OptimizeAcqfGenerator
            acqf = EAVC

            [EAVC]
            target = 0.75

            [QuxGen]
            class = OptimizeAcqfGenerator
            acqf = qLogNoisyExpectedImprovement

            [IndependentGPsModel]
            models = [model1, model2]

            [model1]
            class = GPClassificationModel

            [model2]
            class = GPRegressionModel
        """
        config = Config(config_str=config_str)
        strat = SequentialStrategy.from_config(config)
        while not strat.finished:
            points = strat.gen(1)

            for point in points:
                baz_response = torch.bernoulli(f_2d(point))
                qux_response = f_2d(point, target=torch.tensor([-0.5, -0.5]))

                strat.add_data(point, torch.tensor([[baz_response, qux_response]]))

        x_grid = dim_grid(lower=strat.lb, upper=strat.ub)
        pred_y = strat.model.predict(x_grid)

        norm = torch.distributions.Normal(0, 1)
        baz_target = f_2d(x_grid[np.argmin((norm.cdf(pred_y[0][:, 0]) - 0.75) ** 2)])
        qux_max = x_grid[torch.argmax(pred_y[0][:, 1])]

        # Pretty wide check on binary, but that's the nature of it
        self.assertLessEqual(torch.abs(baz_target - 0.75), 0.15)
        self.assertTrue(torch.all(torch.abs(qux_max - -0.5) < 0.1))


if __name__ == "__main__":
    unittest.main()
