import itertools
import logging
from copy import deepcopy

import aepsych.utils_logging as utils_logging
import multiprocess.context as ctx
import pathos
import torch
from aepsych.benchmark import Benchmark, BenchmarkLogger

ctx._force_start_method("spawn")  # fixes problems with CUDA and fork

logger = utils_logging.getLogger(logging.INFO)


class PathosBenchmark(Benchmark):
    def __init__(self, nproc=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.get_num_threads() > 1:
            logger.warn(
                f"Running benchmark with pytorch threads {torch.get_num_threads()}>1! \n"
                + "Interaction of threaded pytorch with process-based parallelism may be unpredictable!"
            )
        self.pool = pathos.pools.ProcessPool(nodes=nproc)

    def __del__(self):
        # destroy the pool (for when we're testing or running
        # multiple benchmarks in one script) but if the GC already
        # cleared the underlying multiprocessing object (usually on
        # the final call), don't do anything.
        if hasattr(self, "pool"):
            try:
                self.pool.close()
                self.pool.join()
                self.pool.clear()
            except TypeError:
                pass

    def run_experiment(self, config_dict, seed, rep):
        # copy things that we mutate
        local_config = deepcopy(config_dict)
        local_logger = BenchmarkLogger(log_every=self.logger.log_every)
        _ = super().run_experiment(local_config, local_logger, seed, rep)
        return local_logger

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        if "pool" in self_dict.keys():
            del self_dict["pool"]
        if "futures" in self_dict.keys():
            del self_dict["futures"]
        return self_dict

    def run_benchmarks(self):
        self.start_benchmarks()
        self.collate_benchmarks(wait=True)

    def start_benchmarks(self):
        self.loggers = []
        self.all_sim_configs = [
            (config_dict, seed, rep)
            for seed, (config_dict, rep) in enumerate(
                itertools.product(self.combinations, range(self.n_reps))
            )
        ]
        self.futures = [
            self.pool.apipe(self.run_experiment, *conf) for conf in self.all_sim_configs
        ]

    def collate_benchmarks(self, wait=False):
        newfutures = []
        while self.futures:
            item = self.futures.pop()
            if wait or item.ready():
                self.loggers.append(item.get())
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
