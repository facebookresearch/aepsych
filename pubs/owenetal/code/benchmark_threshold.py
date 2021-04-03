# imports
from aepsych.benchmark.test_functions import (
    make_songetal_testfun,
    novel_detection_testfun,
    novel_discrimination_testfun,
)
from aepsych.benchmark import (
    Problem,
    LSEProblem,
    BenchmarkLogger,
    PathosBenchmark,
    combine_benchmarks,
)
from copy import copy
from itertools import product
import os
import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
nproc = 94

n_reps = 100
sobol_trials = 5
total_trials = 150
global_seed = 3
refit_every = 10
log_every = 5

# test functions and boundaries
novel_names = ["novel_detection", "novel_discrimination"]
novel_testfuns = [novel_detection_testfun, novel_discrimination_testfun]
novel_bounds = [{"lb": [-1, -1], "ub": [1, 1]}, {"lb": [-1, -1], "ub": [1, 1]}]
song_phenotypes = ["Metabolic", "Sensory", "Metabolic+Sensory", "Older-normal"]
song_betavals = [0.2, 0.5, 1, 2, 5, 10]
song_testfuns = [
    make_songetal_testfun(p, b) for p, b in product(song_phenotypes, song_betavals)
]
song_bounds = [{"lb": [-3, -20], "ub": [4, 120]}] * len(song_testfuns)
song_names = [f"song_p{p}_b{b}" for p, b in product(song_phenotypes, song_betavals)]
all_testfuns = song_testfuns + novel_testfuns
all_bounds = song_bounds + novel_bounds
all_names = song_names + novel_names

combo_logger = BenchmarkLogger(log_every=log_every)

# benchmark configs, have to subdivide into 5
# configs Sobol, MCLSETS, and Song vs ours get set up all differently
# Song benches
bench_config_nonsobol_song = {
    "common": {"pairwise": False, "target": 0.75},
    "experiment": {
        "acqf": [
            "MCLevelSetEstimation",
            "BernoulliMCMutualInformation",
            "MCPosteriorVariance",
        ],
        "modelbridge_cls": "SingleProbitModelbridgeWithSongHeuristic",
        "init_strat_cls": "SobolStrategy",
        "opt_strat_cls": "EpsilonGreedyModelWrapperStrategy",
        "model": "GPClassificationModel",
        "parnames": "[context,intensity]",
    },
    "MCLevelSetEstimation": {
        "target": 0.75,
        "beta": 3.98,
        "objective": "ProbitObjective",
    },
    "GPClassificationModel": {
        "inducing_size": 100,
        "dim": 2,
        "mean_covar_factory": ["song_mean_covar_factory",],
    },
    "SingleProbitModelbridgeWithSongHeuristic": {"restarts": 10, "samps": 1000},
    "SobolStrategy": {"n_trials": [sobol_trials],},
    "EpsilonGreedyModelWrapperStrategy": {
        "n_trials": [total_trials - sobol_trials],
        "refit_every": [refit_every],
    },
}
bench_config_sobol_song = {
    "common": {"pairwise": False, "target": 0.75},
    "experiment": {
        "acqf": "MCLevelSetEstimation",
        "modelbridge_cls": "SingleProbitModelbridgeWithSongHeuristic",
        "init_strat_cls": "SobolStrategy",
        "opt_strat_cls": "EpsilonGreedyModelWrapperStrategy",
        "model": "GPClassificationModel",
        "parnames": "[context,intensity]",
    },
    "MCLevelSetEstimation": {
        "target": 0.75,
        "beta": 3.98,
        "objective": "ProbitObjective",
    },
    "GPClassificationModel": {
        "inducing_size": 100,
        "dim": 2,
        "mean_covar_factory": ["song_mean_covar_factory",],
    },
    "SingleProbitModelbridgeWithSongHeuristic": {"restarts": 10, "samps": 1000},
    "SobolStrategy": {
        "n_trials": list(range(sobol_trials, total_trials - 1, log_every)),
    },
    "EpsilonGreedyModelWrapperStrategy": {
        "n_trials": [1],
        "refit_every": [refit_every],
    },
}
# non-Song benches

bench_config_sobol_rbf = {
    "common": {"pairwise": False, "target": 0.75},
    "experiment": {
        "acqf": "MonotonicMCLSE",
        "modelbridge_cls": "MonotonicSingleProbitModelbridge",
        "init_strat_cls": "SobolStrategy",
        "opt_strat_cls": "EpsilonGreedyModelWrapperStrategy",
        "model": "MonotonicGPLSETS",
        "parnames": "[context,intensity]",
    },
    "MonotonicMCLSE": {"target": 0.75, "beta": 3.98,},
    "MonotonicGPLSETS": {
        "inducing_size": 100,
        "mean_covar_factory": ["monotonic_mean_covar_factory",],
        "monotonic_idxs": ["[1]", "[]"],
        "uniform_idxs": "[]",
    },
    "MonotonicSingleProbitModelbridge": {"restarts": 10, "samps": 1000},
    "SobolStrategy": {
        "n_trials": list(range(sobol_trials, total_trials - 1, log_every)),
    },
    "EpsilonGreedyModelWrapperStrategy": {
        "n_trials": [1],
        "refit_every": [refit_every],
    },
}
bench_config_all_but_gplsets_rbf = {
    "common": {"pairwise": False, "target": 0.75},
    "experiment": {
        "acqf": [
            "MonotonicMCLSE",
            "MonotonicBernoulliMCMutualInformation",
            "MonotonicMCPosteriorVariance",
        ],
        "modelbridge_cls": "MonotonicSingleProbitModelbridge",
        "init_strat_cls": "SobolStrategy",
        "opt_strat_cls": "EpsilonGreedyModelWrapperStrategy",
        "model": "MonotonicGP",
        "parnames": "[context,intensity]",
    },
    "MonotonicMCLSE": {"target": 0.75, "beta": 3.98,},
    "MonotonicBernoulliMCMutualInformation": {},
    "MonotonicMCPosteriorVariance": {},
    "MonotonicGP": {
        "inducing_size": 100,
        "mean_covar_factory": ["monotonic_mean_covar_factory",],
        "monotonic_idxs": ["[1]", "[]"],
        "uniform_idxs": "[]",
    },
    "MonotonicSingleProbitModelbridge": {"restarts": 10, "samps": 1000},
    "SobolStrategy": {"n_trials": [sobol_trials],},
    "EpsilonGreedyModelWrapperStrategy": {
        "n_trials": [total_trials - sobol_trials],
        "refit_every": [refit_every],
    },
}
bench_config_gplsets_rbf = {
    "common": {"pairwise": False, "target": 0.75},
    "experiment": {
        "acqf": "MonotonicMCLSE",
        "modelbridge_cls": "MonotonicSingleProbitModelbridge",
        "init_strat_cls": "SobolStrategy",
        "opt_strat_cls": "EpsilonGreedyModelWrapperStrategy",
        "model": "MonotonicGPLSETS",
        "parnames": "[context,intensity]",
    },
    "MonotonicMCLSE": {"target": 0.75, "beta": 3.98,},
    "MonotonicGPLSETS": {
        "inducing_size": 100,
        "mean_covar_factory": ["monotonic_mean_covar_factory",],
        "monotonic_idxs": ["[1]", "[]"],
        "uniform_idxs": "[]",
    },
    "MonotonicSingleProbitModelbridge": {"restarts": 10, "samps": 1000},
    "SobolStrategy": {"n_trials": [sobol_trials],},
    "EpsilonGreedyModelWrapperStrategy": {
        "n_trials": [total_trials - sobol_trials],
        "refit_every": [refit_every],
    },
}
all_bench_configs = [
    bench_config_sobol_song,
    bench_config_nonsobol_song,
    bench_config_sobol_rbf,
    bench_config_all_but_gplsets_rbf,
    bench_config_gplsets_rbf,
]


def make_problemobj(testfun, lb, ub):
    # This constructs a Problem from a
    # test function and bounds

    class Inner(LSEProblem, Problem):
        def f(self, x):
            return testfun(x)

    obj = Inner(lb=lb, ub=ub)

    return obj


def make_bench(testfun, logger, name, configs, lb, ub):
    # make a bench object from test function config
    # and bench config
    benches = []
    problem = make_problemobj(testfun, lb, ub)
    for config in configs:
        full_config = copy(config)
        full_config["common"]["lb"] = str(lb)
        full_config["common"]["ub"] = str(ub)
        full_config["common"]["name"] = name
        benches.append(
            PathosBenchmark(
                nproc=nproc,
                problem=problem,
                logger=logger,
                configs=full_config,
                global_seed=global_seed,
                n_reps=n_reps,
            )
        )
    return combine_benchmarks(*benches)


if __name__ == "__main__":
    all_benchmarks = [
        make_bench(testfun, combo_logger, name, all_bench_configs, **bounds)
        for (testfun, bounds, name) in zip(all_testfuns, all_bounds, all_names)
    ]

    for bench in tqdm.tqdm(all_benchmarks):
        print(f"starting {bench}...")
        bench.run_benchmarks()
        print(f"done with {bench}...")

    for bench in all_benchmarks:
        combo_logger._log.extend(bench.logger._log)

    out_pd = combo_logger.pandas()

    out_pd.to_csv(f"bench_{global_seed}.csv")
