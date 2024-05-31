# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import os
import logging
import argparse

# run each job single-threaded, paralellize using pathos
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# multi-socket friendly args
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"
import torch

# force torch to 1 thread too just in case
torch.set_num_interop_threads(1)
torch.set_num_threads(1)

import time
from copy import deepcopy
from pathlib import Path

from aepsych.benchmark import run_benchmarks_with_checkpoints
import aepsych.utils_logging as utils_logging
logger=utils_logging.getLogger(logging.ERROR)

from aepsych.benchmark.problem import (
    DiscrimLowDim,
    DiscrimHighDim,
    Hartmann6Binary,
    ContrastSensitivity6d,  # This takes a few minutes to instantiate due to fitting the model
)

problem_map = {
    "discrim_lowdim": DiscrimLowDim,
    "discrim_highdim": DiscrimHighDim,
    "hartmann6_binary": Hartmann6Binary,
    "contrast_sensitivity_6d": ContrastSensitivity6d,
}


def make_argparser():
    parser = argparse.ArgumentParser(description="Lookahead LSE Benchmarks")
    parser.add_argument("--nproc", type=int, default=30)
    parser.add_argument("--reps_per_chunk", type=int, default=20)
    parser.add_argument("--acqf_start_idx", type=int, default=0)
    parser.add_argument("--sobol_start_idx", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=15)
    parser.add_argument("--opt_size", type=int, default=740) # 490
    parser.add_argument("--init_size", type=int, default=10)
    parser.add_argument("--global_seed", type=int, default=1000)
    parser.add_argument("--log_frequency", type=int, default=10)
    parser.add_argument("--output_path", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--bench_name", type=str, default="exploration_baseline")
    parser.add_argument(
        "--problem",
        type=str,
        choices=[
            "discrim_highdim",
            "discrim_lowdim",
            "hartmann6_binary",
            "contrast_sensitivity_6d",
            "all",
        ],
        default="all",
    )
    return parser


if __name__ == "__main__":

    parser = make_argparser()
    args = parser.parse_args()
    chunks = args.chunks # The number of chunks to break the results into. Each chunk will contain at least 1 run of every
    # combination of problem and config.
    acqf_start_idx = args.acqf_start_idx  # The index of the first chunk to run for different acquisition functions
    sobol_start_idx = args.sobol_start_idx  # The index of the first chunk to run for sobol sampling
    reps_per_chunk = args.reps_per_chunk  # Number of repetitions to run each problem/config in each chunk.

    nproc = args.nproc  # how many processes to use
    global_seed = args.global_seed  # random seed for reproducibility
    log_every = args.log_frequency  # log to csv every this many trials
    checkpoint_every = 120  # save intermediate results every this many seconds
    serial_debug = False  # whether to run simulations serially for debugging
    bench_name=args.bench_name

    out_fname_base = args.output_path
    out_fname_base.mkdir(
        parents=True, exist_ok=True
    )  # make an output folder if not exist
    if args.problem == "all":
        problems = [
            DiscrimLowDim(),
            DiscrimHighDim(),
            Hartmann6Binary(),
            ContrastSensitivity6d(),
        ]
    else:
        problems = [problem_map[args.problem]()]

    bench_config = {
        "common": {
            "stimuli_per_trial": 1,
            "outcome_types": "binary",
            "strategy_names": "[init_strat, opt_strat]",
        },
        "init_strat": {"n_trials": args.init_size, "generator": "SobolGenerator"},
        "opt_strat": {
            "model": "GPClassificationModel",
            "generator": "OptimizeAcqfGenerator",
            "n_trials": args.opt_size,
            "refit_every": args.log_frequency,
        },
        "GPClassificationModel": {
            "inducing_size": 100,
            "mean_covar_factory": "default_mean_covar_factory",
            "inducing_point_method": "auto",
        },
        "default_mean_covar_factory": {
            "fixed_mean": False,
            "lengthscale_priout_fname_baseor": "gamma",
            "outputscale_prior": "gamma",
            "kernel": "RBFKernel",
        },
        "OptimizeAcqfGenerator": {
            "acqf": [
                "MCPosteriorVariance",  # BALV
                "BernoulliMCMutualInformation",  # BALD
            ],
            "restarts": 2,
            "samps": 100,
        },
        # Add the probit transform for non-probit-specific acqfs
        "BernoulliMCMutualInformation": {"objective": "ProbitObjective"},
        "MCPosteriorVariance": {"objective": "ProbitObjective"},
    }

    # benchmaking with baseline acquisition functions
    run_benchmarks_with_checkpoints(
        out_fname_base,
        bench_name,
        problems,
        bench_config,
        global_seed,
        acqf_start_idx,
        chunks,
        reps_per_chunk,
        log_every,
        checkpoint_every,
        nproc,
        serial_debug,
    )

    # benchmaking with sobol sampling
    sobol_config=deepcopy(bench_config)
    sobol_config["opt_strat"]['generator']='SobolGenerator'
    del sobol_config["OptimizeAcqfGenerator"]
    sobol_bench_name=bench_name+"_sobol"
    run_benchmarks_with_checkpoints(
        out_fname_base,
        sobol_bench_name,
        problems,
        sobol_config,
        global_seed,
        sobol_start_idx,
        chunks,
        reps_per_chunk,
        log_every,
        checkpoint_every,
        nproc,
        serial_debug,
    )
