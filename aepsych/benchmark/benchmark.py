#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
import time
import warnings
from copy import deepcopy
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from aepsych.config import Config
from aepsych.strategy import SequentialStrategy
from tqdm.contrib.itertools import product as tproduct

from .logger import BenchmarkLogger
from .problem import Problem


class Benchmark:
    """
    Benchmark base class.

    This class wraps standard functionality for benchmarking models including
    generating cartesian products of run configurations, running the simulated
    experiment loop, and logging results.

    TODO make a benchmarking tutorial and link/refer to it here.
    """

    def __init__(
        self,
        problem: Problem,
        logger: BenchmarkLogger,
        configs: Mapping[str, Union[str, list]],
        global_seed: Optional[int] = None,
        n_reps: int = 1,
    ) -> None:
        """Initialize benchmark.

        Args:
            problem (Problem): Problem object containing the test function to evaluate.
            logger (BenchmarkLogger): BenchmarkLogger object for collecting data from runs.
            configs (Mapping[str, Union[str, list]]): Dictionary of configs to run.
                Lists at leaves are used to construct a cartesian product of configurations.
            global_seed (int, optional): Global seed to use for reproducible benchmarks.
                Defaults to randomized seeds.
            n_reps (int, optional): Number of repetitions to run of each configuration. Defaults to 1.
        """
        self.problem = problem
        self.logger = logger
        self.n_reps = n_reps
        self.combinations = self.make_benchmark_list(**configs)

        if global_seed is None:
            # explicit cast because int and np.int_ are different types
            self.global_seed = int(np.random.randint(0, 200))
        else:
            self.global_seed = global_seed

    def make_benchmark_list(self, **bench_config) -> List[Dict[str, float]]:
        """Generate a list of benchmarks to run from configuration.

            This constructs a cartesian product of config dicts using lists at
            the leaves of the base config

        Returns:
            List[dict[str, float]]: List of dictionaries, each of which can be passed
                to aepsych.config.Config.
        """
        # This could be a generator but then we couldn't
        # know how many params we have, tqdm wouldn't work, etc,
        # so we materialize the full list.
        def gen_combinations(d):
            keys, values = d.keys(), d.values()
            # only go cartesian on list leaves
            values = [v if type(v) == list else [v] for v in values]
            combinations = itertools.product(*values)

            return [dict(zip(keys, c)) for c in combinations]

        keys, values = bench_config.keys(), bench_config.values()
        return [
            dict(zip(keys, c))
            for c in itertools.product(*(gen_combinations(v) for v in values))
        ]

    @property
    def num_benchmarks(self) -> int:
        """Return the total number of runs in this benchmark.

        Returns:
            int: Total number of runs in this benchmark.
        """
        return len(self.combinations) * self.n_reps

    def make_strat_and_flatconfig(
        self, config_dict: Mapping[str, str]
    ) -> Tuple[SequentialStrategy, Dict[str, str]]:
        """From a config dict, generate a strategy (for running) and
            flattened config (for logging)

        Args:
            config_dict (Mapping[str, str]): A run configuration dictionary.

        Returns:
            Tuple[SequentialStrategy, Dict[str,str]]: A tuple containing a strategy
                object and a flat config for the logger.
        """
        config = Config()
        config.update(config_dict=config_dict)
        strat = SequentialStrategy.from_config(config)
        flatconfig = self.flatten_config(config)
        return strat, flatconfig

    def run_experiment(
        self,
        config_dict: Mapping[str, str],
        logger: BenchmarkLogger,
        seed: int,
        rep: int,
    ) -> Tuple[BenchmarkLogger, Optional[SequentialStrategy]]:
        """Run one simulated experiment.

        Args:
            config_dict (Mapping[str, str]): AEPsych configuration to use.
            logger (BenchmarkLogger): BenchmarkLogger object to store data.
            seed (int): Random seed for this run.
            rep (int): Index of this repetition (used for the logger).

        Returns:
            SequentialStrategy: Strategy as of the end of the simulated experiment.
                This is ignored in large-scale benchmarks but useful for one-off visualization.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        strat, flatconfig = self.make_strat_and_flatconfig(config_dict)
        total_gentime = 0.0
        i = 0
        while not strat.finished:
            starttime = time.time()
            next_x = strat.gen()
            gentime = time.time() - starttime
            total_gentime = total_gentime + gentime
            strat.add_data(next_x, [self.problem.sample_y(next_x)])
            if logger.log_at(i) and strat.has_model:
                metrics = self.problem.evaluate(strat)
                logger.log(flatconfig, metrics, i, gentime, rep)

            i = i + 1

        metrics = self.problem.evaluate(strat)
        logger.log(flatconfig, metrics, i, total_gentime, rep, final=True)
        return logger, strat

    def run_benchmarks(self):
        """Run all the benchmarks, sequentially."""
        for i, (rep, config) in enumerate(
            tproduct(range(self.n_reps), self.combinations)
        ):
            local_seed = i + self.global_seed
            _ = self.run_experiment(config, self.logger, seed=local_seed, rep=rep)

    def flatten_config(self, config: Config) -> Dict[str, str]:
        """Flatten a config object for logging.

        Args:
            config (Config): AEPsych config object.

        Returns:
            Dict[str,str]: A flat dictionary that can be used in a
                logger (which underlyingly builds a flat pandas data frame).
        """
        flatconfig = {}
        for s in config.sections():
            flatconfig.update({f"{s}_{k}": v for k, v in config[s].items()})
        return flatconfig

    def __add__(self, bench_2: Benchmark) -> Benchmark:
        """Combine this with another benchmark.

        Benchmarks have to match on loggers, seeds, reps, and problems,
        i.e. only differ in configuration.

        Args:
            bench_2 (Benchmark): Other benchmark to combine.

        Returns:
            Benchmark: Benchmark object containing both sets of configurations.
        """
        assert self.logger == bench_2.logger, (
            f"Cannot combine benchmarks that have different loggers, "
            f"logger1={self.logger}, "
            f"logger2={bench_2.logger}"
        )
        assert self.n_reps == bench_2.n_reps, (
            f"Cannot combine benchmarks that have different n_reps, "
            f"n_reps1={self.n_reps}, "
            f"n_reps2={bench_2.n_reps}"
        )
        assert self.global_seed == bench_2.global_seed, (
            f"Cannot combine benchmarks that have different global_seed, "
            f"global_seed1={self.global_seed}, "
            f"global_seed2={bench_2.global_seed}"
        )
        assert self.problem == bench_2.problem, (
            f"Cannot combine benchmarks that have different problems, "
            f"problem1={self.problem}, "
            f"problem2={bench_2.problem}"
        )

        bench_combined = deepcopy(self)
        bench_combined.combinations.extend(bench_2.combinations)
        return bench_combined


def combine_benchmarks(*benchmarks) -> Benchmark:
    """Combine a set of benchmarks into one.

    Note that this is much less safe than only adding two benchmarks,
    since there is less verification that the benchmarks have the same
    problem, logger, seed, etc. Instead, the Logger and Problem from
    the first benchmark are used.

    Returns:
        Benchmark: A combined benchmark.
    """
    if len(benchmarks) == 1:
        warnings.warn("Calling combine_benchmarks with only one benchmark!")
        return benchmarks[0]
    else:
        bench_combo = benchmarks[0] + benchmarks[1]
        for bench in benchmarks[2:]:
            tmp_bench = deepcopy(bench)
            tmp_bench.logger = bench_combo.logger
            tmp_bench.problem = bench_combo.problem
            bench_combo = bench_combo + tmp_bench
    return bench_combo
