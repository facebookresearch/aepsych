#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import time
import traceback
from copy import deepcopy
from pathlib import Path
from random import shuffle
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import aepsych.utils_logging as utils_logging
import multiprocess.context as ctx
import numpy as np
import pathos
import torch
from aepsych.benchmark import Benchmark
from aepsych.benchmark.problem import Problem
from aepsych.strategy import SequentialStrategy

ctx._force_start_method("spawn")  # fixes problems with CUDA and fork

logger = utils_logging.getLogger(logging.INFO)


class PathosBenchmark(Benchmark):
    """Benchmarking class for parallelized benchmarks using pathos"""

    def __init__(self, nproc: int = 1, *args, **kwargs) -> None:
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

    def __del__(self) -> None:
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
        problem: Problem,
        config_dict: Dict[str, Any],
        seed: int,
        rep: int,
    ) -> Tuple[List[Dict[str, Any]], Union[SequentialStrategy, None]]:
        """Run one simulated experiment.

        Args:
            config_dict (Dict[str, Any]): AEPsych configuration to use.
            seed (int): Random seed for this run.
            rep (int): Index of this repetition.

        Returns:
            Tuple[List[Dict[str, Any]], SequentialStrategy]: A tuple containing a log of the results and the strategy as
                of the end of the simulated experiment. This is ignored in large-scale benchmarks but useful for
                one-off visualization.
        """

        # copy things that we mutate
        local_config = deepcopy(config_dict)
        try:
            return super().run_experiment(problem, local_config, seed, rep)
        except Exception as e:
            logging.error(
                f"Error on config {config_dict}: {e}!"
                + f"Traceback follows:\n{traceback.format_exc()}"
            )

            return [], SequentialStrategy([])

    def __getstate__(self) -> Dict[str, Any]:
        self_dict = self.__dict__.copy()
        if "pool" in self_dict.keys():
            del self_dict["pool"]
        if "futures" in self_dict.keys():
            del self_dict["futures"]
        return self_dict

    def run_benchmarks(self) -> None:
        """Run all the benchmarks,

        Note that this blocks while waiting for benchmarks to complete. If you
        would like to start benchmarks and periodically collect partial results,
        use start_benchmarks and then call collate_benchmarks(wait=False) on some
        interval.
        """
        self.start_benchmarks()
        self.collate_benchmarks(wait=True)

    def start_benchmarks(self) -> None:
        """Start benchmark run.

        This does not block: after running it, self.futures holds the
        status of benchmarks running in parallel.
        """

        def run_discard_strat(*conf) -> List[Dict[str, Any]]:
            logger, _ = self.run_experiment(*conf)
            return logger

        self.all_sim_configs = [
            (problem, config_dict, self.seed + seed, rep)
            for seed, (problem, config_dict, rep) in enumerate(
                itertools.product(self.problems, self.combinations, range(self.n_reps))
            )
        ]
        shuffle(self.all_sim_configs)
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
                results = item.get()
                # filter out empty results from invalid configs
                results = [r for r in results if r != {}]
                if isinstance(results, list):
                    self._log.extend(results)
            else:
                newfutures.append(item)

        self.futures = newfutures


def run_benchmarks_with_checkpoints(
    out_path: str,
    benchmark_name: str,
    problems: List[Problem],
    configs: Mapping[str, Union[str, list]],
    global_seed: Optional[int] = None,
    start_idx: int = 0,
    n_chunks: int = 1,
    n_reps_per_chunk: int = 1,
    log_every: Optional[int] = None,
    checkpoint_every: int = 60,
    n_proc: int = 1,
    serial_debug: bool = False,
) -> None:
    """Runs a series of benchmarks, saving both final and intermediate results to .csv files. Benchmarks are run in
    sequential chunks, each of which runs all combinations of problems/configs/reps in parallel. This function should
    always be used using the "if __name__ == '__main__': ..." idiom.

    Args:
        out_path (str): The path to save the results to.
        benchmark_name (str): A name give to this set of benchmarks. Results will be saved in files named like
            "out_path/benchmark_name_chunk{chunk_number}_out.csv"
        problems (List[Problem]): Problem objects containing the test function to evaluate.
        configs (Mapping[str, Union[str, list]]): Dictionary of configs to run.
            Lists at leaves are used to construct a cartesian product of configurations.
        global_seed (int, optional): Global seed to use for reproducible benchmarks.
            Defaults to randomized seeds.
        start_idx (int): The chunk number to start from after the last checkpoint. Defaults to 0.
        n_chunks (int): The number of chunks to break the results into. Each chunk will contain at least 1 run of every
            combination of problem and config.
        n_reps_per_chunk (int, optional): Number of repetitions to run each problem/config in each chunk.
        log_every (int, optional): Logging interval during an experiment. Defaults to only logging at the end.
        checkpoint_every (int): Save intermediate results every checkpoint_every seconds.
        n_proc (int): Number of processors to use.
        serial_debug: debug serially?
    """
    Path(out_path).mkdir(
        parents=True, exist_ok=True
    )  # make an output folder if not exist
    if serial_debug:
        out_fname = Path(f"{out_path}/{benchmark_name}_out.csv")
        print(f"Starting {benchmark_name} benchmark (serial debug mode)...")
        bench = Benchmark(
            problems=problems,
            configs=configs,
            seed=global_seed,
            n_reps=n_reps_per_chunk * n_chunks,
            log_every=log_every,
        )
        bench.run_benchmarks()
        final_results = bench.pandas()
        final_results.to_csv(out_fname)
    else:
        for chunk in range(start_idx, n_chunks + start_idx):
            out_fname = Path(f"{out_path}/{benchmark_name}_chunk{chunk}_out.csv")

            intermediate_fname = Path(
                f"{out_path}/{benchmark_name}_chunk{chunk}_checkpoint.csv"
            )
            print(f"Starting {benchmark_name} benchmark... chunk {chunk} ")

            bench = PathosBenchmark(
                nproc=n_proc,
                problems=problems,
                configs=configs,
                seed=None,
                n_reps=n_reps_per_chunk,
                log_every=log_every,
            )

            if global_seed is None:
                global_seed = int(np.random.randint(0, 200))
            bench.seed = (
                global_seed + chunk * bench.num_benchmarks
            )  # HACK. TODO: make num_benchmarks a property of bench configs
            bench.start_benchmarks()

            while not bench.is_done:
                time.sleep(checkpoint_every)
                collate_start = time.time()
                print(
                    f"Checkpointing {benchmark_name} chunk {chunk}..., {len(bench.futures)}/{bench.num_benchmarks} alive"
                )
                bench.collate_benchmarks(wait=False)
                temp_results = bench.pandas()
                if len(temp_results) > 0:
                    temp_results["rep"] = temp_results["rep"] + n_reps_per_chunk * chunk
                    temp_results.to_csv(intermediate_fname)
                print(
                    f"Collate done in {time.time()-collate_start} seconds, {len(bench.futures)}/{bench.num_benchmarks} left"
                )

            print(f"{benchmark_name} chunk {chunk} fully done!")
            final_results = bench.pandas()
            final_results["rep"] = final_results["rep"] + n_reps_per_chunk * chunk
            final_results.to_csv(out_fname)
