#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import uuid

import numpy as np
import numpy.testing as npt
import torch
from aepsych.server import AEPsychServer
from aepsych.server.message_handlers.handle_ask import ask
from aepsych.server.message_handlers.handle_setup import configure
from aepsych.server.message_handlers.handle_tell import tell
from gpytorch.likelihoods import GaussianLikelihood

# run on single threads to keep us from deadlocking weirdly in CI
if "CI" in os.environ or "SANDCASTLE" in os.environ:
    torch.set_num_threads(1)


class GPRegressionTest(unittest.TestCase):
    def f(self, x):
        return x**3 - 4 * x**2 + np.random.normal() * 0.1

    def simulate_response(self, trial_params):
        x = trial_params["par1"][0]
        response = self.f(x)
        return response

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)

        dbname = "./{}.db".format(str(uuid.uuid4().hex))
        config = """
            [common]
            parnames = [par1]
            lb = [-1]
            ub = [3]
            stimuli_per_trial=1
            outcome_types=[continuous]
            strategy_names = [init_strat, opt_strat]

            [init_strat]
            min_asks = 10
            generator = SobolGenerator

            [opt_strat]
            min_asks = 5
            generator = OptimizeAcqfGenerator
            model = GPRegressionModel
            acqf = qNoisyExpectedImprovement

            [GPRegressionModel]
            likelihood = GaussianLikelihood
            max_fit_time = 1
            dim = 1
        """
        self.server = AEPsychServer(database_path=dbname)
        configure(self.server, config_str=config)

        while not self.server.strat.finished:
            trial_params = ask(self.server)
            outcome = self.simulate_response(trial_params)
            tell(self.server, outcome, trial_params)

    def tearDown(self):
        self.server.db.delete_db()

    def test_extremum(self):
        tol = 0.2  # don't need to be super precise because it's small data
        fmax, argmax = self.server.strat.get_max()

        npt.assert_allclose(fmax, 0, atol=tol)
        npt.assert_allclose(argmax, 0, atol=tol)

        fmin, argmin = self.server.strat.get_min()
        npt.assert_allclose(fmin, -256 / 27, atol=tol)
        npt.assert_allclose(argmin, 8 / 3, atol=tol)

        val, arg = self.server.strat.inv_query(0)
        npt.assert_allclose(val, 0, atol=tol)
        npt.assert_allclose(arg, 0, atol=tol)

    def test_from_config(self):
        model = self.server.strat.model
        generator = self.server.strat.generator
        npt.assert_allclose(model.transforms.untransform(generator.lb), [-1.0])
        npt.assert_allclose(model.transforms.untransform(generator.ub), [3.0])
        self.assertIsInstance(model.likelihood, GaussianLikelihood)
        self.assertEqual(model.max_fit_time, 1)


if __name__ == "__main__":
    unittest.main()
