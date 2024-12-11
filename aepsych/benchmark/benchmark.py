#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
import time
from random import shuffle
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from aepsych.config import Config
from aepsych.strategy import ensure_model_is_fresh, SequentialStrategy
from tqdm.contrib.itertools import product as tproduct

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
        problems: List[Problem],
        configs: Mapping[str, Union[str, list]],
        seed: Optional[int] = None,
        n_reps: int = 1,
        log_every: Optional[int] = 10,
    ) -> None:
        """Initialize benchmark.

        Args:
            problems (List[Problem]): Problem objects containing the test function to evaluate.
            configs (Mapping[str, Union[str, list]]): Dictionary of configs to run.
                Lists at leaves are used to construct a cartesian product of configurations.
            seed (int, optional): Random seed to use for reproducible benchmarks.
                Defaults to randomized seeds.
            n_reps (int, optional): Number of repetitions to run of each configuration. Defaults to 1.
            log_every (int, optional): Logging interval during an experiment. Defaults to logging every 10 trials.
        """
        self.problems = problems
        self.n_reps = n_reps
        self.combinations = self.make_benchmark_list(**configs)
        self._log: List[Dict[str, object]] = []
        self.log_every = log_every

        # shuffle combinations so that intermediate results have a bit of everything
        shuffle(self.combinations)

        if seed is None:
            # explicit cast because int and np.int_ are different types
            self.seed = int(np.random.randint(0, 200))
        else:
            self.seed = seed

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

    def materialize_config(self, config_dict):
        materialized_config = {}
        for key, value in config_dict.items():
            materialized_config[key] = {
                k: v._evaluate(config_dict) if isinstance(v, DerivedValue) else v
                for k, v in value.items()
            }
        return materialized_config

    @property
    def num_benchmarks(self) -> int:
        """Return the total number of runs in this benchmark.

        Returns:
            int: Total number of runs in this benchmark.
        """
        return len(self.problems) * len(self.combinations) * self.n_reps

    def make_strat_and_flatconfig(
        self, config_dict: Mapping[str, str]
    ) -> Tuple[SequentialStrategy, Dict[str, str]]:
        """From a config dict, generate a strategy (for running) and
            flattened config (for logging)

        Args:
            config_dict (Mapping[str, str]): A run configuration dictionary.

        Returns:
            Tuple[SequentialStrategy, Dict[str,str]]: A tuple containing a strategy
                object and a flat config.
        """
        config = Config()
        config.update(config_dict=config_dict)
        strat = SequentialStrategy.from_config(config)
        flatconfig = self.flatten_config(config)
        return strat, flatconfig

    def run_experiment(
        self,
        problem: Problem,
        config_dict: Dict[str, Any],
        seed: int,
        rep: int,
    ) -> Tuple[List[Dict[str, Any]], Union[SequentialStrategy, None]]:
        """Run one simulated experiment.

        Args:
            config_dict (Dict[str, str]): AEPsych configuration to use.
            seed (int): Random seed for this run.
            rep (int): Index of this repetition.

        Returns:
            Tuple[List[Dict[str, object]], SequentialStrategy]: A tuple containing a log of the results and the strategy as
                of the end of the simulated experiment. This is ignored in large-scale benchmarks but useful for
                one-off visualization.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        config_dict["common"]["lb"] = str(problem.lb.tolist())
        config_dict["common"]["ub"] = str(problem.ub.tolist())
        config_dict["common"]["dim"] = str(problem.lb.shape[0])
        config_dict["common"]["parnames"] = str(
            [f"par{i}" for i in range(len(problem.ub.tolist()))]
        )
        config_dict["problem"] = problem.metadata
        materialized_config = self.materialize_config(config_dict)

        # no-op config
        is_invalid = materialized_config["common"].get("invalid_config", False)
        if is_invalid:
            return [{}], None

        strat, flatconfig = self.make_strat_and_flatconfig(materialized_config)

        problem_metadata = {
            f"problem_{key}": value for key, value in problem.metadata.items()
        }

        total_gentime = 0.0
        total_fittime = 0.0
        i = 0
        results = []
        while not strat.finished:
            starttime = time.time()
            next_x = strat.gen()
            gentime = time.time() - starttime
            total_gentime += gentime
            next_y = problem.sample_y(next_x)
            strat.add_data(next_x, next_y)
            # strat usually defers model fitting until it is needed
            # (e.g. for gen or predict) so that we don't refit
            # unnecessarily. But for benchmarking we want to time
            # fit and gen separately, so we force a strat update
            # so we can time fit vs gen. TODO make this less awkward
            starttime = time.time()
            ensure_model_is_fresh(lambda x: None)(strat._strat)
            fittime = time.time() - starttime
            total_fittime += fittime
            if (self.log_at(i) or strat.finished) and strat.has_model:
                metrics = problem.evaluate(strat)
                result: Dict[str, Union[float, str]] = {
                    "fit_time": fittime,
                    "cum_fit_time": total_fittime,
                    "gen_time": gentime,
                    "cum_gen_time": total_gentime,
                    "trial_id": i,
                    "rep": rep,
                    "seed": seed,
                    "final": strat.finished,
                    "strat_idx": strat._strat_idx,
                }
                result.update(problem_metadata)
                result.update(flatconfig)
                result.update(metrics)
                results.append(result)

            i = i + 1

        return results, strat

    def run_benchmarks(self) -> None:
        """Run all the benchmarks, sequentially."""
        for i, (rep, config, problem) in enumerate(
            tproduct(range(self.n_reps), self.combinations, self.problems)
        ):
            local_seed = i + self.seed
            results, _ = self.run_experiment(problem, config, seed=local_seed, rep=rep)
            if results != [{}]:
                self._log.extend(results)

    def flatten_config(self, config: Config) -> Dict[str, str]:
        """Flatten a config object for logging.

        Args:
            config (Config): AEPsych config object.

        Returns:
            Dict[str,str]: A flat dictionary (that can be used to build a flat pandas data frame).
        """
        flatconfig = {}
        for s in config.sections():
            flatconfig.update({f"{s}_{k}": v for k, v in config[s].items()})
        return flatconfig

    def log_at(self, i: int) -> bool:
        """Check if we should log on this trial index.

        Args:
            i (int): Trial index to (maybe) log at.

        Returns:
            bool: True if this trial should be logged.
        """
        if self.log_every is not None:
            return i % self.log_every == 0
        else:
            return False

    def pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self._log)


class DerivedValue(object):
    """
    A class for dynamically generating config values from other config values during benchmarking.
    """

    def __init__(self, args: List[Tuple[str, str]], func: Callable) -> None:
        """Initialize DerivedValue.

        Args:
            args (List[Tuple[str]]): Each tuple in this list is a pair of strings that refer to keys in a nested dictionary.
            func (Callable): A function that accepts args as input.

        For example, consider the following:

            benchmark_config = {
                "common": {
                    "model": ["GPClassificationModel", "FancyNewModelToBenchmark"],
                    "acqf": "MCLevelSetEstimation"
                },
                "init_strat": {
                    "min_asks": [10, 20],
                    "generator": "SobolGenerator"
                },
                "opt_strat": {
                    "generator": "OptimizeAcqfGenerator",
                    "min_asks":
                        DerivedValue(
                            [("init_strat", "min_asks"), ("common", "model")],
                            lambda x,y : 100 - x if y == "GPClassificationModel" else 50 - x)
                }
            }

        Four separate benchmarks would be generated from benchmark_config:
            1. model = GPClassificationModel; init trials = 10; opt trials = 90
            2. model = GPClassificationModel; init trials = 20; opt trials = 80
            3. model = FancyNewModelToBenchmark; init trials = 10; opt trials = 40
            4. model = FancyNewModelToBenchmark; init trials = 20; opt trials = 30

        Note that if you can also access problem names into func by including ("problem", "name") in args.
        """
        self.args = args
        self.func = func

    def _evaluate(self, benchmark_config: Dict) -> Any:
        """Fetches values of self.args from benchmark_config and evaluates self.func on them."""
        _args = [benchmark_config[outer][inner] for outer, inner in self.args]
        return self.func(*_args)
