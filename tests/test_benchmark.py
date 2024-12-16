#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time
import unittest

import numpy as np
import torch
from aepsych.benchmark import (
    Benchmark,
    DerivedValue,
    example_problems,
    LSEProblem,
    PathosBenchmark,
    Problem,
)
from aepsych.models import GPClassificationModel
from aepsych.models.inducing_points import GreedyVarianceReduction

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def f(x, delay=False):
    if delay:
        time.sleep(0.1 * random.random())
    if len(x.shape) == 1:
        return x
    else:
        return x.sum(axis=-1)


class TestProblem(Problem):
    name = "test problem"
    bounds = np.c_[0, 1].T
    threshold = 0.75

    def f(self, x):
        return f(x)


class TestSlowProblem(TestProblem):
    name = "test slow problem"

    def f(self, x):
        return f(x, delay=True)


class LSETestProblem(LSEProblem):
    name = "test lse problem"
    bounds = np.c_[[-1, -1], [1, 1]].T

    def __init__(self, thresholds=None):
        thresholds = 0.75 if thresholds is None else thresholds
        super().__init__(thresholds=thresholds)

    def f(self, x):
        return f(x)


class MultipleLSETestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        np.random.seed(1)

        self.n_thresholds = 5
        self.thresholds = torch.linspace(0.55, 0.95, self.n_thresholds)
        self.test_problem = example_problems.DiscrimLowDim(thresholds=self.thresholds)

        self.model = GPClassificationModel(
            dim=2,
            inducing_point_method=GreedyVarianceReduction(dim=2),
        )

    def unvectorized_p_below_threshold(self, x, f_thresh) -> torch.Tensor:
        """this is the original p_below_threshold method in the AEPsychMixin that calculates model prediction
        of the probability of the stimulus being below a threshold
        for one single threshold"""
        f, var = self.model.predict(x)

        # Perform all operations in PyTorch (no .detach().numpy())
        z = (f_thresh - f) / var.sqrt()
        return torch.distributions.Normal(0, 1).cdf(z)  # Use PyTorch's CDF equivalent

    def unvectorized_true_below_threshold(self, threshold):
        """the original true_below_threshold method in the LSEProblem class"""
        return (self.test_problem.p(self.test_problem.eval_grid) <= threshold).to(
            torch.float32
        )

    def test_vectorized_score_calculation(self):
        f_thresholds = self.test_problem.f_threshold(self.model)
        p_l = self.model.p_below_threshold(self.test_problem.eval_grid, f_thresholds)
        true_p_l = self.test_problem.true_below_threshold
        # Now, perform the Brier score calculation and classification error in PyTorch
        brier_p_below_thresh = torch.mean(2 * torch.square(true_p_l - p_l), dim=1)
        # Classification error
        misclass_on_thresh = torch.mean(
            p_l * (1 - true_p_l) + (1 - p_l) * true_p_l, dim=1
        )
        assert (
            p_l.ndim == 2
            and p_l.shape == true_p_l.shape
            and p_l.shape[0] == len(self.thresholds)
        )

        for i_threshold, single_threshold in enumerate(self.thresholds):
            normal_dist = torch.distributions.Normal(0, 1)
            single_f_threshold = normal_dist.icdf(
                single_threshold
            ).float()  # equivalent to norm.ppf

            assert torch.isclose(single_f_threshold, f_thresholds[i_threshold])

            unvectorized_p_l = self.unvectorized_p_below_threshold(
                self.test_problem.eval_grid, single_f_threshold
            )
            assert torch.all(torch.isclose(unvectorized_p_l, p_l[i_threshold]))

            unvectorized_true_p_l = self.unvectorized_true_below_threshold(
                single_threshold
            )
            assert torch.all(
                torch.isclose(unvectorized_true_p_l, true_p_l[i_threshold])
            )

            unvectorized_brier_score = torch.mean(
                2 * torch.square(unvectorized_true_p_l - unvectorized_p_l)
            )
            assert torch.isclose(
                unvectorized_brier_score, brier_p_below_thresh[i_threshold]
            )

            unvectorized_misclass_err = torch.mean(
                unvectorized_p_l * (1 - unvectorized_true_p_l)
                + (1 - unvectorized_p_l) * unvectorized_true_p_l
            )
            assert torch.isclose(
                unvectorized_misclass_err, misclass_on_thresh[i_threshold]
            )


class BenchmarkTestCase(unittest.TestCase):
    def setUp(self):
        # run this single-threaded since we parallelize using pathos
        self.oldenv = os.environ.copy()
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_MAX_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        os.environ["KMP_BLOCKTIME"] = "1"

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.bench_config = {
            "common": {
                "invalid_config": DerivedValue(
                    [("init_strat", "min_asks")],
                    lambda min_asks: True if min_asks > 2 else False,
                ),
                "stimuli_per_trial": 1,
                "outcome_types": ["binary"],
                "strategy_names": "[init_strat, opt_strat]",
            },
            "experiment": {
                "acqf": "MCLevelSetEstimation",
                "model": "GPClassificationModel",
            },
            "init_strat": {
                "min_asks": [2, 4],
                "generator": "SobolGenerator",
                "min_total_outcome_occurrences": 0,
            },
            "opt_strat": {
                "min_asks": [
                    DerivedValue(
                        [("problem", "name")], lambda x: 1 + int(x == "test problem")
                    ),
                    DerivedValue(
                        [("problem", "name")], lambda x: 2 + int(x == "test problem")
                    ),
                ],
                "generator": "OptimizeAcqfGenerator",
                "min_total_outcome_occurrences": 0,
            },
            "MCLevelSetEstimation": {
                "target": 0.75,
                "beta": 3.84,
            },
            "GPClassificationModel": {
                "inducing_size": 10,
                "mean_covar_factory": "default_mean_covar_factory",
                "refit_every": 100,
                "max_fit_time": 0.1,
            },
            "OptimizeAcqfGenerator": {
                "restarts": 1,
                "samps": 20,
                "max_gen_time": 0.1,
            },
        }

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.oldenv)

    def test_bench_smoke(self):
        problem1 = TestProblem()
        problem2 = LSETestProblem()

        bench = Benchmark(
            problems=[problem1, problem2],
            configs=self.bench_config,
            n_reps=2,
            log_every=2,
        )
        bench.run_benchmarks()

        out = bench.pandas()

        # assert problem metadata was correctly saved
        self.assertEqual(
            sorted(out["problem_name"].unique()), ["test lse problem", "test problem"]
        )
        self.assertEqual(
            sorted(
                out[out["problem_name"] == "test lse problem"][
                    "problem_thresholds"
                ].unique()
            ),
            ["0.75"],
        )

        # assert derived values work correctly
        self.assertEqual(
            sorted(
                out[out["problem_name"] == "test problem"][
                    "opt_strat_min_asks"
                ].unique()
            ),
            ["2", "3"],
        )
        self.assertEqual(
            sorted(
                out[out["problem_name"] == "test lse problem"][
                    "opt_strat_min_asks"
                ].unique()
            ),
            ["1", "2"],
        )

        # have as many final results as we expect. Because of invalid trials,
        # only half of benchmarks are valid
        self.assertTrue(len(out[out.final]) == bench.num_benchmarks // 2)

        # have as many repetitions as we expect
        self.assertTrue(len(out.rep.unique()) == bench.n_reps)

        # reporting intervals are correct
        self.assertTrue((out[~out.final].trial_id % 2 == 0).all())

        # we don't run extra trials
        total_trials = out.init_strat_min_asks.astype(
            int
        ) + out.opt_strat_min_asks.astype(int)
        self.assertTrue((out.trial_id <= total_trials).all())

        # ensure each simulation has a unique random seed
        self.assertTrue(out[out["final"]]["seed"].is_unique)

    def test_bench_pathossmoke(self):
        problem1 = TestProblem()
        problem2 = LSETestProblem()

        bench = PathosBenchmark(
            problems=[problem1, problem2], configs=self.bench_config, n_reps=2, nproc=2
        )
        bench.run_benchmarks()
        out = bench.pandas()

        # assert problem metadata was correctly saved
        self.assertEqual(
            sorted(out["problem_name"].unique()), ["test lse problem", "test problem"]
        )
        self.assertEqual(
            sorted(
                out[out["problem_name"] == "test lse problem"][
                    "problem_thresholds"
                ].unique()
            ),
            ["0.75"],
        )

        # assert derived values work correctly
        self.assertEqual(
            sorted(
                out[out["problem_name"] == "test problem"][
                    "opt_strat_min_asks"
                ].unique()
            ),
            ["2", "3"],
        )
        self.assertEqual(
            sorted(
                out[out["problem_name"] == "test lse problem"][
                    "opt_strat_min_asks"
                ].unique()
            ),
            ["1", "2"],
        )

        # have as many final results as we expect (half of configs are invalid)
        self.assertTrue(len(out[out.final]) == bench.num_benchmarks // 2)

        # have as many repetitions as we expect
        self.assertTrue(len(out.rep.unique()) == bench.n_reps)

        # reporting intervals are correct
        self.assertTrue((out[~out.final].trial_id % 2 == 0).all())

        # we don't run extra trials
        total_trials = out.init_strat_min_asks.astype(
            int
        ) + out.opt_strat_min_asks.astype(int)
        self.assertTrue((out.trial_id <= total_trials).all())

        # ensure each simulation has a unique random seed
        self.assertTrue(out[out["final"]]["seed"].is_unique)

    def test_bench_pathos_partial(self):
        """
        test that we can launch async and get partial results
        """
        problem = TestSlowProblem()

        bench = PathosBenchmark(
            problems=[problem], configs=self.bench_config, n_reps=1, log_every=2
        )
        bench.start_benchmarks()
        # wait for something to finish
        while len(bench._log) == 0:
            time.sleep(0.1)
            bench.collate_benchmarks(wait=False)

        out = bench.pandas()  # this should only be a partial result
        # have fewer than all the results (which is half of all benchmarks
        # since half are invalid)
        self.assertTrue(len(out[out.final]) < (bench.num_benchmarks // 2))

        bench.collate_benchmarks(wait=True)  # wait for everything to finish
        out = bench.pandas()  # complete results

        # now we should have everything (valid = half of all benchmarks)
        self.assertTrue(len(out[out.final]) == (bench.num_benchmarks // 2))


class BenchProblemTestCase(unittest.TestCase):
    def setUp(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)

    def test_nonmonotonic_single_lse_eval(self):
        config = {
            "common": {
                "stimuli_per_trial": 1,
                "outcome_types": ["binary"],
                "strategy_names": "[init_strat, opt_strat]",
                "acqf": "MCLevelSetEstimation",
                "model": "GPClassificationModel",
            },
            "init_strat": {"generator": "SobolGenerator", "min_asks": 50},
            "opt_strat": {"generator": "OptimizeAcqfGenerator", "min_asks": 1},
            "MCLevelSetEstimation": {
                "target": 0.75,
                "beta": 3.84,
            },
            "GPClassificationModel": {
                "inducing_size": 10,
                "mean_covar_factory": "default_mean_covar_factory",
            },
            "OptimizeAcqfGenerator": {
                "restarts": 10,
                "samps": 1000,
            },
        }
        problem = LSETestProblem()
        bench = Benchmark(problems=[problem], configs=config, log_every=100)
        _, strat = bench.run_experiment(problem, bench.combinations[0], 0, 0)
        e = problem.evaluate(strat)
        self.assertTrue(e["mean_square_err_p"] < 0.05)

    def test_monotonic_single_lse_eval(self):
        config = {
            "common": {
                "stimuli_per_trial": 1,
                "outcome_types": ["binary"],
                "strategy_names": "[init_strat, opt_strat]",
                "acqf": "MonotonicMCLSE",
                "model": "MonotonicRejectionGP",
            },
            "init_strat": {"generator": "SobolGenerator", "min_asks": 50},
            "opt_strat": {"generator": "MonotonicRejectionGenerator", "min_asks": 1},
            "SobolGenerator": {"seed": 1},
            "MonotonicMCLSE": {
                "target": 0.75,
                "beta": 3.84,
            },
            "MonotonicRejectionGP": {
                "inducing_size": 10,
                "mean_covar_factory": "monotonic_mean_covar_factory",
                "monotonic_idxs": "[1]",
            },
            "MonotonicRejectionGenerator": {
                "model_gen_options": {
                    "num_restarts": 10,
                    "raw_samples": 1000,
                }
            },
        }
        problem = LSETestProblem()
        bench = Benchmark(problems=[problem], configs=config, log_every=100)
        _, strat = bench.run_experiment(problem, bench.combinations[0], 0, 0)

        e = problem.evaluate(strat)
        self.assertTrue(e["mean_square_err_p"] < 0.05)


if __name__ == "__main__":
    unittest.main()
