# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import logging
import unittest

import aepsych.utils_logging as utils_logging
import numpy as np
import torch
from aepsych.benchmark import Benchmark, example_problems
from aepsych.utils import make_scaled_sobol
from scipy.stats import norm

logger = utils_logging.getLogger(logging.ERROR)


class TestPairwise(unittest.TestCase):
    def setUp(self) -> None:
        self.problem_map = {
            "PairwiseHartmann6Binary": "Hartmann6Binary",
            "PairwiseDiscrimHighdim": "DiscrimHighDim",
            "PairwiseDiscrimLowdim": "DiscrimLowDim",
        }

        self.problems = {}
        for pairwise_problem, single_problem in self.problem_map.items():
            self.problems[pairwise_problem] = getattr(
                example_problems, pairwise_problem
            )()
            self.problems[single_problem] = getattr(example_problems, single_problem)()

    def test_pairwise_probability(self) -> None:
        for pairwise_problem, single_problem in self.problem_map.items():
            pairwise_problem = self.problems[pairwise_problem]
            single_problem = self.problems[single_problem]

            x1, x2 = make_scaled_sobol(single_problem.lb, single_problem.ub, 2)
            pairwise_x = torch.concat([x1, x2]).unsqueeze(0)

            pairwise_p = pairwise_problem.p(pairwise_x)
            f1 = single_problem.f(x1.unsqueeze(0))
            f2 = single_problem.f(x2.unsqueeze(0))
            single_p = norm.cdf(f1 - f2)
            self.assertTrue(np.allclose(pairwise_p, single_p))

    def pairwise_benchmark_smoketest(self) -> None:
        """a smoke test to make sure the models and benchmark are set up correctly"""
        config = {
            "common": {
                "stimuli_per_trial": 2,
                "outcome_types": "binary",
                "strategy_names": "[init_strat, opt_strat]",
            },
            "init_strat": {"n_trials": 10, "generator": "SobolGenerator"},
            "opt_strat": {
                "model": "GPClassificationModel",
                "generator": "SobolGenerator",
                "n_trials": 10,
                "refit_every": 10,
            },
            "GPClassificationModel": {
                "inducing_size": 100,
                "mean_covar_factory": "default_mean_covar_factory",
                "inducing_point_method": "auto",
            },
        }

        pairwise_problems = [self.problems[name] for name in self.problem_map.keys()]

        bench = Benchmark(
            problems=pairwise_problems,
            configs=config,
            n_reps=1,
        )
        bench.run_benchmarks()


if __name__ == "__main__":
    unittest.main()
