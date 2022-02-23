#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import traceback
from copy import deepcopy
from typing import Mapping, Tuple, Optional

import aepsych.utils_logging as utils_logging
import multiprocess.context as ctx
import pathos
import torch
from aepsych.benchmark import Benchmark, BenchmarkLogger
from aepsych.strategy import SequentialStrategy


ctx._force_start_method("spawn")  # fixes problems with CUDA and fork

logger = utils_logging.getLogger(logging.INFO)


class PathosBenchmark(Benchmark):
    """Benchmarking class for parallelized benchmarks using pathos"""

    def __init__(self, nproc: int = 1, *args, **kwargs):
        """Initialize pathos benchmark.

        Args:
            nproc (int, optional): Number of cores to use. Defaults to 1.
        """
        super().__init__(*args, **kwargs)

        # parallelize over jobs, so each job should be 1 thread only
        num_threads = torch.get_num_threads()
        num_interopt_threads = torch.get_num_interop_threads()
        if num_threads > 1 or num_interopt_threads > 1:
            raise RuntimeError(
                "PathosBenchmark parallelizes over threads,"
                + "and as such is incompatible with torch being threaded. "
                + "Please call `torch.set_num_threads(1)` and "
                + "`torch.set_num_interop_threads(1)` before using PathosBenchmark!"
            )
        cores_available = pathos.multiprocessing.cpu_count()
        if nproc >= cores_available:
            raise RuntimeError(
                f"Requesting a benchmark with {nproc} cores but "
                + f"machine has {cores_available} cores! It is highly "
                "recommended to leave at least 1-2 cores open for OS tasks."
            )
        self.pool = pathos.pools.ProcessPool(nodes=nproc)

    def __del__(self):
        # destroy the pool (for when we're testing or running
        # multiple benchmarks in one script) but if the GC already
        # cleared the underlying multiprocessing object (usually on
        # the final call), don't do anything.
        if hasattr(self, "pool") and self.pool is not None:
            try:
                self.pool.close()
                self.pool.join()
                self.pool.clear()
            except TypeError:
                pass

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
            BenchmarkLogger: Local logger from this run.
        """

        # copy things that we mutate
        local_config = deepcopy(config_dict)
        local_logger = deepcopy(logger)
        try:
            return super().run_experiment(local_config, local_logger, seed, rep)
        except Exception as e:

            logging.error(
                f"Error on config {config_dict}: {e}!"
                + f"Traceback follows:\n{traceback.format_exc()}"
            )

            return local_logger, None

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        if "pool" in self_dict.keys():
            del self_dict["pool"]
        if "futures" in self_dict.keys():
            del self_dict["futures"]
        return self_dict

    def run_benchmarks(self):
        """Run all the benchmarks,

        Note that this blocks while waiting for benchmarks to complete. If you
        would like to start benchmarks and periodically collect partial results,
        use start_benchmarks and then call collate_benchmarks(wait=False) on some
        interval.
        """
        self.start_benchmarks()
        self.collate_benchmarks(wait=True)

    def start_benchmarks(self):
        """Start benchmark run.

        This does not block: after running it, self.futures holds the
        status of benchmarks running in parallel.
        """

        def run_discard_strat(*conf):
            logger, _ = self.run_experiment(*conf)
            return logger

        self.loggers = []
        self.all_sim_configs = [
            (config_dict, self.logger, self.global_seed + seed, rep)
            for seed, (config_dict, rep) in enumerate(
                itertools.product(self.combinations, range(self.n_reps))
            )
        ]
        self.futures = [
            self.pool.apipe(run_discard_strat, *conf) for conf in self.all_sim_configs
        ]

    @property
    def is_done(self) -> bool:
        """Check if the benchmark is done.

        Returns:
            bool: True if all futures are cleared and benchmark is done.
        """
        return len(self.futures) == 0

    def collate_benchmarks(self, wait: bool = False) -> None:
        """Collect benchmark results from completed futures.

        Args:
            wait (bool, optional): If true, this method blocks and waits
            on all futures to complete. Defaults to False.
        """
        newfutures = []
        while self.futures:
            item = self.futures.pop()
            if wait or item.ready():
                result = item.get()
                if isinstance(result, BenchmarkLogger):
                    self.loggers.append(result)
            else:
                newfutures.append(item)

        self.futures = newfutures

        if len(self.loggers) > 0:
            out_logger = BenchmarkLogger()
            for logger in self.loggers:
                out_logger._log.extend(logger._log)
            self.logger = out_logger

    def __add__(self, bench_2):
        out = super().__add__(bench_2)
        out.pool = self.pool
        return out
