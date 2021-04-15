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
    combine_benchmarks,
    Benchmark,
    PathosBenchmark,
    Problem,
    LSEProblem,
    BenchmarkLogger,
)


class TestProblem(Problem, LSEProblem):
    def f(self, x, delay=False):
        if delay:
            time.sleep(1 * random.random())
        if len(x.shape) == 1:
            return x
        else:
            return x.sum(axis=-1)


class BenchmarkTestCase(unittest.TestCase):
    def setUp(self):

        # run this single-threaded since we parallelize using pathos
        self.oldenv = os.environ.copy()
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_MAX_THREADS"] = "1"
        os.environ["MKL_THREADING_LAYER"] = "GNU"
        os.environ["MKL_NUM_THREADS"] = "1"

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.bench_config = {
            "common": {
                "lb": "[0]",
                "ub": "[1]",
                "outcome_type": "single_probit",
            },
            "experiment": {
                "acqf": "LevelSetEstimation",
                "modelbridge_cls": "SingleProbitModelbridge",
                "init_strat_cls": "SobolStrategy",
                "opt_strat_cls": "ModelWrapperStrategy",
            },
            "LevelSetEstimation": {
                "target": 0.75,
                "beta": 3.98,
            },
            "GPClassificationModel": {
                "inducing_size": 10,
                "mean_covar_factory": "default_mean_covar_factory",
            },
            "SingleProbitModelbridge": {
                "restarts": 10,
                "samps": 1000,
            },
            "SobolStrategy": {
                "n_trials": [2, 4, 6],
            },
            "ModelWrapperStrategy": {
                "n_trials": [1, 2],
            },
        }

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.oldenv)

    def test_bench_smoke(self):

        logger = BenchmarkLogger(log_every=2)
        problem = TestProblem(lb=[0], ub=[1])

        bench = Benchmark(
            problem=problem, logger=logger, configs=self.bench_config, n_reps=2
        )
        bench.run_benchmarks()

        out = bench.logger.pandas()

        # have as many final results as we expect
        self.assertTrue(len(out[out.final]) == bench.num_benchmarks)

        # have as many repetitions as we expect
        self.assertTrue(len(out.rep.unique()) == bench.n_reps)

        # reporting intervals are correct
        self.assertTrue((out[~out.final].trial_id % 2 == 0).all())

        # we don't run extra trials
        total_trials = out.SobolStrategy_n_trials.astype(
            int
        ) + out.ModelWrapperStrategy_n_trials.astype(int)
        self.assertTrue((out.trial_id <= total_trials).all())

    def test_bench_pathossmoke(self):

        logger = BenchmarkLogger(log_every=2)
        problem = TestProblem(lb=[0], ub=[1])

        bench = PathosBenchmark(
            problem=problem, logger=logger, configs=self.bench_config, n_reps=2
        )
        bench.run_benchmarks()
        out = bench.logger.pandas()

        # have as many final results as we expect
        self.assertTrue(len(out[out.final]) == bench.num_benchmarks)

        # have as many repetitions as we expect
        self.assertTrue(len(out.rep.unique()) == bench.n_reps)

        # reporting intervals are correct
        self.assertTrue((out[~out.final].trial_id % 2 == 0).all())

        # we don't run extra trials
        total_trials = out.SobolStrategy_n_trials.astype(
            int
        ) + out.ModelWrapperStrategy_n_trials.astype(int)
        self.assertTrue((out.trial_id <= total_trials).all())

    def test_bench_pathos_partial(self):
        """
        test that we can launch async and get partial results
        """
        logger = BenchmarkLogger(log_every=2)
        problem = TestProblem(lb=[0], ub=[1], delay=True)

        bench = PathosBenchmark(
            problem=problem, logger=logger, configs=self.bench_config, n_reps=1
        )
        bench.start_benchmarks()
        # wait for something to finsh
        while len(bench.logger._log) == 0:
            time.sleep(0.1)
            bench.collate_benchmarks(wait=False)

        out = bench.logger.pandas()  # this should only be a partial result
        # have fewer than all the results
        self.assertTrue(len(out[out.final]) < bench.num_benchmarks)

        bench.collate_benchmarks(wait=True)  # wait for everything to finsh
        out = bench.logger.pandas()  # complete results

        self.assertTrue(len(out[out.final]) == bench.num_benchmarks)

    def test_add_bench(self):

        logger = BenchmarkLogger(log_every=2)
        problem = TestProblem(lb=[0], ub=[1])

        bench_1 = Benchmark(
            problem=problem,
            logger=logger,
            configs=self.bench_config,
            global_seed=2,
            n_reps=2,
        )
        bench_2 = Benchmark(
            problem=problem,
            logger=logger,
            configs=self.bench_config,
            global_seed=2,
            n_reps=2,
        )

        bench_combined = bench_1 + bench_2
        three_bench = combine_benchmarks(bench_1, bench_2, bench_1)

        self.assertTrue((len(bench_combined.combinations) == 12))
        self.assertTrue((len(three_bench.combinations) == 18))
        self.assertTrue((len(bench_1.combinations) == 6))
        self.assertTrue((len(bench_2.combinations) == 6))


class BenchProblemTestCase(unittest.TestCase):
    def setUp(self):

        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)

    def test_nonmonotonic_single_lse_eval(self):
        config = {
            "common": {
                "lb": "[-1, -1]",
                "ub": "[1, 1]",
                "outcome_type": "single_probit",
            },
            "experiment": {
                "acqf": "LevelSetEstimation",
                "modelbridge_cls": "SingleProbitModelbridge",
                "init_strat_cls": "SobolStrategy",
                "opt_strat_cls": "ModelWrapperStrategy",
            },
            "LevelSetEstimation": {
                "target": 0.75,
                "beta": 3.98,
            },
            "GPClassificationModel": {
                "inducing_size": 10,
                "mean_covar_factory": "default_mean_covar_factory",
            },
            "SingleProbitModelbridge": {
                "restarts": 10,
                "samps": 1000,
            },
            "SobolStrategy": {
                "n_trials": 50,
            },
            "ModelWrapperStrategy": {
                "n_trials": 1,
            },
        }
        problem = TestProblem(lb=[-1, -1], ub=[1, 1])
        logger = BenchmarkLogger(log_every=100)
        bench = Benchmark(problem=problem, configs=config, logger=logger)
        strat = bench.run_experiment(bench.combinations[0], logger, 0, 0)
        e = problem.evaluate(strat)
        self.assertTrue(e["mean_square_err_p"] < 0.05)

    def test_monotonic_single_lse_eval(self):
        config = {
            "common": {
                "lb": "[-1, -1]",
                "ub": "[1, 1]",
                "outcome_type": "single_probit",
            },
            "experiment": {
                "acqf": "MonotonicMCLSE",
                "modelbridge_cls": "MonotonicSingleProbitModelbridge",
                "init_strat_cls": "SobolStrategy",
                "opt_strat_cls": "ModelWrapperStrategy",
                "model": "MonotonicRejectionGP",
            },
            "MonotonicMCLSE": {
                "target": 0.75,
                "beta": 3.98,
            },
            "MonotonicRejectionGP": {
                "inducing_size": 10,
                "mean_covar_factory": "monotonic_mean_covar_factory",
            },
            "MonotonicSingleProbitModelbridge": {
                "restarts": 10,
                "samps": 1000,
            },
            "SobolStrategy": {
                "n_trials": 50,
            },
            "ModelWrapperStrategy": {
                "n_trials": 1,
            },
        }
        problem = TestProblem(lb=[-1, -1], ub=[1, 1])
        logger = BenchmarkLogger(log_every=100)
        bench = Benchmark(problem=problem, configs=config, logger=logger)
        strat = bench.run_experiment(bench.combinations[0], logger, 0, 0)
        e = problem.evaluate(strat)
        self.assertTrue(e["mean_square_err_p"] < 0.05)


if __name__ == "__main__":
    unittest.main()
