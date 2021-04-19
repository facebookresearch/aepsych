#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import time
import warnings
from copy import deepcopy

import numpy as np
import torch
from aepsych.config import Config
from aepsych.strategy import SequentialStrategy
from tqdm.contrib.itertools import product as tproduct


class Benchmark:
    def __init__(self, problem, logger, configs, global_seed=None, n_reps=1):
        self.problem = problem
        self.logger = logger
        self.n_reps = n_reps
        self.combinations = self.make_benchmark_list(**configs)

        if global_seed is None:
            self.global_seed = np.random.randint(0, 200)
        else:
            self.global_seed = global_seed

    def make_benchmark_list(self, **bench_config):
        # yeah this could be a generator but then we couldn't
        # know how many params we have, tqdm wouldn't work, etc
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
    def num_benchmarks(self):
        return len(self.combinations) * self.n_reps

    def make_strat_and_flatconfig(self, config_dict):
        config = Config()
        config.update(config_dict=config_dict)
        strat = SequentialStrategy.from_config(config)
        flatconfig = self.flatten_config(config)
        return strat, flatconfig

    def run_experiment(self, config_dict, logger, seed, rep):
        torch.manual_seed(seed)
        np.random.seed(seed)
        strat, flatconfig = self.make_strat_and_flatconfig(config_dict)
        total_gentime = 0
        i = 0
        while not strat.finished:
            starttime = time.time()
            next_x = strat.gen()
            gentime = time.time() - starttime
            total_gentime = total_gentime + gentime
            strat.add_data(next_x, [self.problem.sample_y(next_x)])
            if logger.log_at(i) and strat.has_model:
                metrics = self.problem.evaluate(strat)
                logger.log(strat, flatconfig, metrics, i, gentime, rep)

            i = i + 1

        metrics = self.problem.evaluate(strat)
        logger.log(strat, flatconfig, metrics, i, total_gentime, rep, final=True)
        return strat  # ignored in strat but useful for visualizing posterior one-off

    def run_benchmarks(self):
        for i, (rep, config) in enumerate(
            tproduct(range(self.n_reps), self.combinations)
        ):
            local_seed = i + self.global_seed
            _ = self.run_experiment(config, self.logger, seed=local_seed, rep=rep)

    def flatten_config(self, config):
        flatconfig = {}
        for s in config.sections():
            flatconfig.update({f"{s}_{k}": v for k, v in config[s].items()})
        return flatconfig

    def __add__(self, bench_2):
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


def combine_benchmarks(*benchmarks):
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
