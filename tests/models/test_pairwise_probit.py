#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
import uuid

import numpy as np
import numpy.testing as npt
import torch
from aepsych import server, utils_logging
from aepsych.acquisition.objective import ProbitObjective
from aepsych.benchmark.test_functions import f_1d, f_2d, f_pairwise, new_novel_det
from aepsych.config import Config
from aepsych.generators import OptimizeAcqfGenerator, SobolGenerator
from aepsych.models import PairwiseProbitModel
from aepsych.server.message_handlers.handle_ask import ask
from aepsych.server.message_handlers.handle_setup import configure
from aepsych.server.message_handlers.handle_tell import tell
from aepsych.strategy import SequentialStrategy, Strategy
from aepsych.transforms import (
    ParameterTransformedGenerator,
    ParameterTransformedModel,
    ParameterTransforms,
)
from aepsych.transforms.ops import NormalizeScale
from aepsych.utils import dim_grid
from botorch.acquisition import qUpperConfidenceBound
from botorch.acquisition.active_learning import PairwiseMCPosteriorVariance
from scipy.stats import bernoulli, norm, pearsonr


class PairwiseProbitModelStrategyTest(unittest.TestCase):
    def test_pairs_to_comparisons(self):
        def ptc_numpy(x, y, dim):
            """
            old numpy impl of pairs to comparisons
            """

            # This needs to take a unique over the feature dim by flattening
            # over pairs but not instances/batches. This is actually tensor
            # matricization over the feature dimension but awkward in numpy
            unique_coords = np.unique(np.moveaxis(x, 1, 0).reshape(dim, -1), axis=1)

            def _get_index_of_equal_row(arr, x, axis=0):
                return np.argwhere(np.all(np.equal(arr, x[:, None]), axis=axis)).item()

            comparisons = []
            for pair, judgement in zip(x, y):
                comparison = (
                    _get_index_of_equal_row(unique_coords, pair[..., 0]),
                    _get_index_of_equal_row(unique_coords, pair[..., 1]),
                )
                if judgement == 0:
                    comparisons.append(comparison)
                else:
                    comparisons.append(comparison[::-1])
            return torch.Tensor(unique_coords.T), torch.LongTensor(comparisons)

        x = np.random.normal(size=(10, 1, 2))
        y = np.random.choice((0, 1), size=10)

        datapoints1, comparisons1 = ptc_numpy(x, y, 1)

        pbo = PairwiseProbitModel(lb=[-10], ub=[10])
        datapoints2, comparisons2 = pbo._pairs_to_comparisons(
            torch.Tensor(x), torch.Tensor(y)
        )
        npt.assert_equal(datapoints1.numpy(), datapoints2.numpy())
        npt.assert_equal(comparisons1.numpy(), comparisons2.numpy())

        x = np.random.normal(size=(10, 2, 2))
        y = np.random.choice((0, 1), size=10)

        datapoints1, comparisons1 = ptc_numpy(x, y, 2)

        pbo = PairwiseProbitModel(lb=[-10], ub=[10], dim=2)
        datapoints2, comparisons2 = pbo._pairs_to_comparisons(
            torch.Tensor(x), torch.Tensor(y)
        )
        npt.assert_equal(datapoints1.numpy(), datapoints2.numpy())
        npt.assert_equal(comparisons1.numpy(), comparisons2.numpy())

    def test_pairwise_probit_batched(self):
        """
        test our 1d gaussian bump example
        """
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 20
        n_opt = 1
        lb = torch.tensor([-4.0, 1e-5])
        ub = torch.tensor([-1e-5, 4.0])
        extra_acqf_args = {"beta": 3.84}
        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed, stimuli_per_trial=2),
                min_asks=n_init,
                stimuli_per_trial=2,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=PairwiseProbitModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    acqf=qUpperConfidenceBound,
                    acqf_kwargs=extra_acqf_args,
                    stimuli_per_trial=2,
                    lb=lb,
                    ub=ub,
                ),
                min_asks=n_opt,
                stimuli_per_trial=2,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(model_list)

        while not strat.finished:
            next_pair = strat.gen(num_points=3)
            # next_pair is batch x dim x pair,
            # this checks that we have the reshapes
            # right
            self.assertTrue((next_pair[:, 0, :] < 0).all())
            self.assertTrue((next_pair[:, 1, :] > 0).all())
            strat.add_data(
                next_pair,
                bernoulli.rvs(
                    f_pairwise(f_1d, next_pair.sum(1), noise_scale=0.1).squeeze()
                ),
            )

        xgrid = dim_grid(lb, ub, gridsize=10)

        zhat, _ = strat.predict(xgrid)
        # true max is 0, very loose test
        self.assertTrue(xgrid[torch.argmax(zhat, 0)].sum().detach().numpy() < 0.5)

    def test_pairwise_memorize(self):
        """
        can we memorize a simple function
        """
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        lb = [-1, -1]
        ub = [1, 1]
        gen = SobolGenerator(lb=lb, ub=ub, seed=seed, stimuli_per_trial=2)
        x = torch.Tensor(gen.gen(num_points=20))
        # "noiseless" new_novel_det (just take the mean instead of sampling)
        y = torch.Tensor(f_pairwise(new_novel_det, x) > 0.5).int()
        model = PairwiseProbitModel(lb=lb, ub=ub)
        model.fit(x[:18], y[:18])
        with torch.no_grad():
            f0, _ = model.predict(x[18:, ..., 0])
            f1, _ = model.predict(x[18:, ..., 1])
            pred_diff = norm.cdf(f1 - f0)
        pred = pred_diff > 0.5
        npt.assert_allclose(pred, y[18:])

    def test_pairwise_memorize_rescaled(self):
        """
        can we memorize a simple function (with rescaled inputs)
        """
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        lb = [-1000, 0]
        ub = [0, 1e-5]
        gen = SobolGenerator(lb=lb, ub=ub, seed=seed, stimuli_per_trial=2)
        x = torch.Tensor(gen.gen(num_points=20))
        # "noiseless" new_novel_det (just take the mean instead of sampling)
        xrescaled = x.clone()
        xrescaled[:, 0, :] = xrescaled[:, 0, :] / 500 + 1
        xrescaled[:, 1, :] = xrescaled[:, 1, :] / 5e-6 - 1
        y = torch.Tensor(f_pairwise(new_novel_det, xrescaled) > 0.5).int()
        model = PairwiseProbitModel(lb=lb, ub=ub)
        model.fit(x[:18], y[:18])
        with torch.no_grad():
            f0, _ = model.predict(x[18:, ..., 0])
            f1, _ = model.predict(x[18:, ..., 1])
            pred_diff = norm.cdf(f1 - f0)
        pred = pred_diff > 0.5
        npt.assert_allclose(pred, y[18:])

    def test_1d_pairwise_probit(self):
        """
        test our 1d gaussian bump example
        """
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        lb = torch.tensor([-4.0])
        ub = torch.tensor([4.0])
        extra_acqf_args = {"beta": 3.84}
        transforms = ParameterTransforms(
            normalize=NormalizeScale(d=1, bounds=torch.stack([lb, ub]))
        )
        sobol_gen = ParameterTransformedGenerator(
            generator=SobolGenerator,
            lb=lb,
            ub=ub,
            seed=seed,
            stimuli_per_trial=2,
            transforms=transforms,
        )
        acqf_gen = ParameterTransformedGenerator(
            generator=OptimizeAcqfGenerator,
            acqf=qUpperConfidenceBound,
            acqf_kwargs=extra_acqf_args,
            stimuli_per_trial=2,
            transforms=transforms,
            lb=lb,
            ub=ub,
        )
        probit_model = ParameterTransformedModel(
            model=PairwiseProbitModel, lb=lb, ub=ub, transforms=transforms
        )
        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                generator=sobol_gen,
                min_asks=n_init,
                stimuli_per_trial=2,
                outcome_types=["binary"],
                transforms=transforms,
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=probit_model,
                generator=acqf_gen,
                min_asks=n_opt,
                stimuli_per_trial=2,
                outcome_types=["binary"],
                transforms=transforms,
            ),
        ]

        strat = SequentialStrategy(model_list)

        for _i in range(n_init + n_opt):
            next_pair = strat.gen()
            strat.add_data(
                next_pair, [bernoulli.rvs(f_pairwise(f_1d, next_pair, noise_scale=0.1))]
            )

        x = torch.linspace(-4, 4, 100)

        zhat, _ = strat.predict(x)
        # true max is 0, very loose test
        self.assertTrue(np.abs(x[np.argmax(zhat.detach().numpy())]) < 0.5)

    def test_1d_pairwise_probit_pure_exploration(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        lb = torch.tensor([-2.0])
        ub = torch.tensor([2.0])

        acqf = PairwiseMCPosteriorVariance
        extra_acqf_args = {"objective": ProbitObjective()}

        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed, stimuli_per_trial=2),
                min_asks=n_init,
                stimuli_per_trial=2,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=PairwiseProbitModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    acqf=acqf,
                    acqf_kwargs=extra_acqf_args,
                    stimuli_per_trial=2,
                    lb=lb,
                    ub=ub,
                ),
                min_asks=n_opt,
                stimuli_per_trial=2,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(model_list)

        for _i in range(n_init + n_opt):
            next_pair = strat.gen()
            strat.add_data(
                next_pair,
                [bernoulli.rvs(f_pairwise(lambda x: x, next_pair, noise_scale=0.1))],
            )

        test_gen = SobolGenerator(lb=lb, ub=ub, seed=seed + 1, stimuli_per_trial=2)
        test_x = torch.Tensor(test_gen.gen(100))

        ftrue_test = (test_x[..., 0] - test_x[..., 1]).squeeze()

        with torch.no_grad():
            fdiff_test = (
                strat.model.predict(test_x[..., 0], rereference=None)[0]
                - strat.model.predict(test_x[..., 1], rereference=None)[0]
            )

        self.assertTrue(pearsonr(fdiff_test, ftrue_test)[0] >= 0.9)

        with torch.no_grad():
            fdiff_test_reref = (
                strat.model.predict(test_x[..., 0])[0]
                - strat.model.predict(test_x[..., 1])[0]
            )

        self.assertTrue(pearsonr(fdiff_test_reref, ftrue_test)[0] >= 0.9)

    def test_2d_pairwise_probit(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 20
        n_opt = 1
        lb = torch.tensor([-1.0, -1.0])
        ub = torch.tensor([1.0, 1.0])
        extra_acqf_args = {"beta": 3.84}

        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed, stimuli_per_trial=2),
                min_asks=n_init,
                stimuli_per_trial=2,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=PairwiseProbitModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    acqf=qUpperConfidenceBound,
                    acqf_kwargs=extra_acqf_args,
                    stimuli_per_trial=2,
                    lb=lb,
                    ub=ub,
                ),
                min_asks=n_opt,
                stimuli_per_trial=2,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(model_list)

        for _i in range(n_init + n_opt):
            next_pair = strat.gen()
            strat.add_data(
                next_pair, [bernoulli.rvs(f_pairwise(f_2d, next_pair, noise_scale=0.1))]
            )

        xy = np.mgrid[-1:1:30j, -1:1:30j].reshape(2, -1).T

        zhat, _ = strat.predict(torch.Tensor(xy))

        # true min is at 0,0
        self.assertTrue(np.all(np.abs(xy[np.argmax(zhat.detach().numpy())]) < 0.2))

    def test_2d_pairwise_probit_pure_exploration(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 20
        n_opt = 1
        lb = torch.tensor([-1.0, -1.0])
        ub = torch.tensor([1.0, 1.0])
        acqf = PairwiseMCPosteriorVariance
        extra_acqf_args = {"objective": ProbitObjective()}

        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed, stimuli_per_trial=2),
                min_asks=n_init,
                stimuli_per_trial=2,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=PairwiseProbitModel(lb=lb, ub=ub),
                generator=OptimizeAcqfGenerator(
                    acqf=acqf,
                    acqf_kwargs=extra_acqf_args,
                    stimuli_per_trial=2,
                    lb=lb,
                    ub=ub,
                ),
                min_asks=n_opt,
                stimuli_per_trial=2,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(model_list)

        for _i in range(n_init + n_opt):
            next_pair = strat.gen()
            strat.add_data(
                next_pair, [bernoulli.rvs(f_pairwise(new_novel_det, next_pair))]
            )

        xy = torch.stack(
            torch.meshgrid(torch.linspace(-1, 1, 30), torch.linspace(-1, 1, 30)), dim=-1
        ).view(-1, 2)

        zhat, _ = strat.predict(xy)

        ztrue = new_novel_det(xy)

        corr = pearsonr(zhat.detach().flatten(), ztrue.flatten())[0]
        self.assertTrue(corr > 0.80)

    def test_sobolmodel_pairwise(self):
        # test that SobolModel correctly gets bounds

        sobol_x = np.zeros((10, 3, 2))
        mod = Strategy(
            lb=[1, 2, 3],
            ub=[2, 3, 4],
            min_asks=10,
            stimuli_per_trial=2,
            outcome_types=["binary"],
            generator=SobolGenerator(
                lb=[1, 2, 3], ub=[2, 3, 4], seed=12345, stimuli_per_trial=2
            ),
        )

        for i in range(10):
            sobol_x[i, ...] = mod.gen()

        self.assertTrue(np.all(sobol_x[:, 0, :] > 1))
        self.assertTrue(np.all(sobol_x[:, 1, :] > 2))
        self.assertTrue(np.all(sobol_x[:, 2, :] > 3))
        self.assertTrue(np.all(sobol_x[:, 0, :] < 2))
        self.assertTrue(np.all(sobol_x[:, 1, :] < 3))
        self.assertTrue(np.all(sobol_x[:, 2, :] < 4))

    def test_hyperparam_consistency(self):
        # verify that creating the model `from_config` or with `__init__` has the same hyperparams

        m1 = PairwiseProbitModel(lb=[1, 2], ub=[3, 4])

        m2 = PairwiseProbitModel.from_config(
            config=Config(
                config_dict={
                    "common": {"lb": "[1,2]", "ub": "[3,4]", "parnames": "[par1, par2]"}
                }
            )
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


class PairwiseProbitModelServerTest(unittest.TestCase):
    def setUp(self):
        # setup logger
        server.logger = utils_logging.getLogger(logging.DEBUG, "logs")
        # random datebase path name without dashes
        database_path = "./{}.db".format(str(uuid.uuid4().hex))
        self.s = server.AEPsychServer(database_path=database_path)

    def tearDown(self):
        self.s.cleanup()

        # cleanup the db
        if self.s.db is not None:
            self.s.db.delete_db()

    def test_1d_pairwise_server(self):
        seed = 123
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 2
        config_str = f"""
            [common]
            lb = [-4]
            ub = [4]
            stimuli_per_trial = 2
            outcome_types =[binary]
            parnames = [x]
            strategy_names = [init_strat, opt_strat]
            acqf = PairwiseMCPosteriorVariance

            [init_strat]
            min_asks = {n_init}
            generator = SobolGenerator

            [opt_strat]
            model = PairwiseProbitModel
            min_asks = {n_opt}
            generator = OptimizeAcqfGenerator

            [PairwiseProbitModel]
            mean_covar_factory = default_mean_covar_factory

            [PairwiseMCPosteriorVariance]
            objective = ProbitObjective

            [OptimizeAcqfGenerator]
            restarts = 10
            samps = 1000
            """

        server = self.s
        configure(
            server,
            config_str=config_str,
        )

        for _i in range(n_init + n_opt):
            next_config = ask(server)

            next_x = torch.tensor(next_config["x"], dtype=torch.float64)

            next_y = bernoulli.rvs(f_pairwise(f_1d, next_x, noise_scale=0.1))
            tell(server, config=next_config, outcome=next_y)

        x = torch.linspace(-4, 4, 100)
        zhat, _ = server.strat.predict(x)
        self.assertTrue(np.abs(x[np.argmax(zhat.detach().numpy())]) < 0.5)

    def test_2d_pairwise_server(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        config_str = f"""
            [common]
            lb = [-1, -1]
            ub = [1, 1]
            stimuli_per_trial=2
            outcome_types=[binary]
            parnames = [x, y]
            strategy_names = [init_strat, opt_strat]
            acqf = PairwiseMCPosteriorVariance

            [init_strat]
            min_asks = {n_init}
            generator = SobolGenerator

            [opt_strat]
            min_asks = {n_opt}
            model = PairwiseProbitModel
            generator = OptimizeAcqfGenerator

            [PairwiseProbitModel]
            mean_covar_factory = default_mean_covar_factory

            [PairwiseMCPosteriorVariance]
            objective = ProbitObjective

            [OptimizeAcqfGenerator]
            restarts = 10
            samps = 1000
            """

        server = self.s
        configure(
            server,
            config_str=config_str,
        )
        for _i in range(n_init + n_opt):
            next_config = ask(server)
            next_pair = torch.stack(
                (torch.tensor(next_config["x"]), torch.tensor(next_config["y"])), dim=1
            )
            next_y = bernoulli.rvs(f_pairwise(f_2d, next_pair, noise_scale=0.1))
            tell(server, config=next_config, outcome=next_y)

        xy = np.mgrid[-1:1:30j, -1:1:30j].reshape(2, -1).T

        zhat, _ = server.strat.predict(torch.Tensor(xy))

        # true min is at 0,0
        self.assertTrue(np.all(np.abs(xy[np.argmax(zhat.detach().numpy())]) < 0.2))

    def test_serialization_1d(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 3
        n_opt = 1
        config_str = f"""
            [common]
            lb = [-4]
            ub = [4]
            stimuli_per_trial=2
            outcome_types=[binary]
            parnames = [x]
            strategy_names = [init_strat, opt_strat]
            acqf = PairwiseMCPosteriorVariance

            [init_strat]
            min_asks = {n_init}
            generator = SobolGenerator

            [opt_strat]
            model = PairwiseProbitModel
            min_asks = {n_opt}
            generator = OptimizeAcqfGenerator

            [PairwiseProbitModel]
            mean_covar_factory = default_mean_covar_factory

            [PairwiseMCPosteriorVariance]
            objective = ProbitObjective

            [OptimizeAcqfGenerator]
            restarts = 10
            samps = 1000
            """

        server = self.s
        configure(server, config_str=config_str)

        for _i in range(n_init + n_opt):
            next_config = ask(server)
            next_x = torch.tensor(next_config["x"], dtype=torch.float64)

            next_y = bernoulli.rvs(f_pairwise(f_1d, next_x))
            tell(server, config=next_config, outcome=next_y)

        import dill

        # just make sure it works
        try:
            s = dill.dumps(server)
            server2 = dill.loads(s)
            self.assertEqual(len(server2._strats), len(server._strats))
            for strat1, strat2 in zip(server._strats, server2._strats):
                self.assertEqual(type(strat1), type(strat2))
                self.assertEqual(
                    type(strat1.model._base_obj), type(strat2.model._base_obj)
                )
                self.assertTrue(torch.equal(strat1.x, strat2.x))
                self.assertTrue(torch.equal(strat1.y, strat2.y))

        except Exception:
            self.fail()

    def test_serialization_2d(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 3
        n_opt = 1

        config_str = f"""
            [common]
            lb = [-1, -1]
            ub = [1, 1]
            stimuli_per_trial=2
            outcome_types=[binary]
            parnames = [x, y]
            strategy_names = [init_strat, opt_strat]
            acqf = PairwiseMCPosteriorVariance

            [init_strat]
            min_asks = {n_init}
            generator = SobolGenerator

            [opt_strat]
            model = PairwiseProbitModel
            min_asks = {n_opt}
            generator = OptimizeAcqfGenerator

            [PairwiseProbitModel]
            mean_covar_factory = default_mean_covar_factory

            [PairwiseMCPosteriorVariance]
            objective = ProbitObjective

            [OptimizeAcqfGenerator]
            restarts = 10
            samps = 1000
            """

        server = self.s

        configure(server, config_str=config_str)

        for _i in range(n_init + n_opt):
            next_config = ask(server)
            next_pair = torch.stack(
                (torch.tensor(next_config["x"]), torch.tensor(next_config["y"])), dim=0
            )
            next_y = bernoulli.rvs(f_pairwise(f_2d, next_pair))
            tell(server, config=next_config, outcome=next_y)

        import dill

        # just make sure it works
        try:
            s = dill.dumps(server)
            server2 = dill.loads(s)
            self.assertEqual(len(server2._strats), len(server._strats))
            for strat1, strat2 in zip(server._strats, server2._strats):
                self.assertEqual(type(strat1), type(strat2))
                self.assertEqual(
                    type(strat1.model._base_obj), type(strat2.model._base_obj)
                )
                self.assertTrue(torch.equal(strat1.x, strat2.x))
                self.assertTrue(torch.equal(strat1.y, strat2.y))
        except Exception:
            self.fail()

    def test_config_to_tensor(self):
        config_str = """
            [common]
            lb = [-1]
            ub = [0]
            stimuli_per_trial=2
            outcome_types=[binary]
            parnames = [x]
            strategy_names = [init_strat, opt_strat]
            acqf = PairwiseMCPosteriorVariance

            [init_strat]
            min_asks = 1
            generator = SobolGenerator

            [opt_strat]
            model = PairwiseProbitModel
            min_asks = 1
            generator = OptimizeAcqfGenerator

            [PairwiseProbitModel]
            mean_covar_factory = default_mean_covar_factory

            [PairwiseMCPosteriorVariance]
            objective = ProbitObjective

            [OptimizeAcqfGenerator]
            restarts = 10
            samps = 1000
            """
        server = self.s

        configure(server, config_str=config_str)

        conf = ask(server)

        self.assertTrue(server._config_to_tensor(conf).shape == (1, 2))

        config_str = """
            [common]
            lb = [-1, 0]
            ub = [0, 1]
            stimuli_per_trial=2
            outcome_types=[binary]
            parnames = [x, y]
            strategy_names = [init_strat, opt_strat]
            acqf = PairwiseMCPosteriorVariance

            [init_strat]
            min_asks = 1
            generator = SobolGenerator

            [opt_strat]
            model = PairwiseProbitModel
            min_asks = 1
            generator = OptimizeAcqfGenerator

            [PairwiseProbitModel]
            mean_covar_factory = default_mean_covar_factory

            [PairwiseMCPosteriorVariance]
            objective = ProbitObjective

            [OptimizeAcqfGenerator]
            restarts = 10
            samps = 1000
            """

        configure(server, config_str=config_str)

        conf = ask(server)

        self.assertTrue(server._config_to_tensor(conf).shape == (2, 2))

        config_str = """
            [common]
            lb = [-1, 1e-6, 10]
            ub = [-1e-6, 1, 100]
            stimuli_per_trial=2
            outcome_types=[binary]
            parnames = [x, y, z]
            strategy_names = [init_strat, opt_strat]
            acqf = PairwiseMCPosteriorVariance

            [init_strat]
            min_asks = 1
            generator = SobolGenerator

            [opt_strat]
            model = PairwiseProbitModel
            min_asks = 1
            generator = OptimizeAcqfGenerator

            [PairwiseProbitModel]
            mean_covar_factory = default_mean_covar_factory

            [PairwiseMCPosteriorVariance]
            objective = ProbitObjective

            [OptimizeAcqfGenerator]
            restarts = 10
            samps = 1000
            """

        configure(server, config_str=config_str)

        conf = ask(server)

        tensor = server._config_to_tensor(conf)
        self.assertTrue(tensor.shape == (3, 2))

        # Check if reshapes were correct
        self.assertTrue(torch.all(tensor[0, :] <= -1e-6))
        self.assertTrue(
            torch.all(torch.logical_and(tensor[1, :] >= 1e-6, tensor[1, :] <= 1))
        )
        self.assertTrue(torch.all(tensor[2, :] >= 10))


if __name__ == "__main__":
    unittest.main()
