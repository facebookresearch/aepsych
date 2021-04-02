# imports
from bayesopt_server.benchmark.test_functions import (
    make_songetal_testfun,
    novel_detection_testfun,
    novel_discrimination_testfun,
)

from bayesopt_server.benchmark import (
    Problem,
    LSEProblem,
    BenchmarkLogger,
    Benchmark,
    combine_benchmarks,
)
from copy import copy
from itertools import product


bench_cls = Benchmark  # use vanilla Benchmark for serial debugging

n_reps = 100
sobol_trials = 20
total_trials = 100
global_seed = 3
refit_every = 10
log_every = 5

# test functions and boundaries
hairtie_names = ["novel_detection", "novel_discrimination"]
hairtie_testfuns = [novel_detection_testfun, novel_discrimination_testfun]
hairtie_bounds = [{"lb": [-1, -1], "ub": [1, 1]}, {"lb": [-1, -1], "ub": [1, 1]}]
# song_phenotypes = ["Metabolic+Sensory", "Older-normal"]
song_phenotypes = ["Metabolic", "Sensory", "Metabolic+Sensory", "Older-normal"]
# song_betavals = [2]
song_betavals = [0.2, 0.5, 1, 2, 5, 10]
song_testfuns = [
    make_songetal_testfun(p, b) for p, b in product(song_phenotypes, song_betavals)
]
song_bounds = [{"lb": [-3, -20], "ub": [4, 120]}] * len(song_testfuns)
song_names = [f"song_p{p}_b{b}" for p, b in product(song_phenotypes, song_betavals)]
all_testfuns = song_testfuns + hairtie_testfuns
all_bounds = song_bounds + hairtie_bounds
all_names = song_names + hairtie_names

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
        "modelbridge_cls": "SingleProbitBayesOptWithSongHeuristic",
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
    "SingleProbitBayesOptWithSongHeuristic": {"restarts": 10, "samps": 1000},
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
        "modelbridge_cls": "SingleProbitBayesOptWithSongHeuristic",
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
    "SingleProbitBayesOptWithSongHeuristic": {"restarts": 10, "samps": 1000},
    "SobolStrategy": {
        "n_trials": list(range(sobol_trials, total_trials - 1, log_every)),
    },
    "EpsilonGreedyModelWrapperStrategy": {"n_trials": [1], "refit_every": [refit_every]},
}
# non-Song benches

bench_config_sobol_rbf = {
    "common": {"pairwise": False, "target": 0.75},
    "experiment": {
        "acqf": "MonotonicMCLSE",
        "modelbridge_cls": "MonotonicSingleProbitBayesOpt",
        "init_strat_cls": "SobolStrategy",
        "opt_strat_cls": "EpsilonGreedyModelWrapperStrategy",
        "model": "MonotonicGPLSETS",
        "parnames": "[context,intensity]",
    },
    "MonotonicMCLSE": {"target": 0.75, "beta": 3.98,},
    "MonotonicGPLSETS": {
        "inducing_size": 100,
        "mean_covar_factory": [
            "conditioning_mean_covar_factory",
            "monotonic_mean_covar_factory",
        ],
        "monotonic_idxs": ["[1]", "[]"],
        "uniform_idxs": "[]",
    },
    "conditioning_mean_covar_factory": {
        "fval_upper": "4",
        "fval_lower": "-4",
        "cond_noise": "0",
        "intensity_dim": "1",
        "num_pinned_induc": "10",
        "base_factory": "monotonic_mean_covar_factory",
    },
    "MonotonicSingleProbitBayesOpt": {"restarts": 10, "samps": 1000},
    "SobolStrategy": {
        "n_trials": list(range(sobol_trials, total_trials - 1, log_every)),
    },
    "EpsilonGreedyModelWrapperStrategy": {"n_trials": [1], "refit_every": [refit_every]},
}
bench_config_all_but_gplsets_rbf = {
    "common": {"pairwise": False, "target": 0.75},
    "experiment": {
        "acqf": [
            "MonotonicMCLSE",
            "MonotonicBernoulliMCMutualInformation",
            "MonotonicMCPosteriorVariance",
        ],
        "modelbridge_cls": "MonotonicSingleProbitBayesOpt",
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
        "mean_covar_factory": [
            "conditioning_mean_covar_factory",
            "monotonic_mean_covar_factory",
        ],
        "monotonic_idxs": ["[1]", "[]"],
        "uniform_idxs": "[]",
    },
    "conditioning_mean_covar_factory": {
        "fval_upper": "4",
        "fval_lower": "-4",
        "cond_noise": "0",
        "intensity_dim": "1",
        "num_pinned_induc": "10",
        "base_factory": "monotonic_mean_covar_factory",
    },
    "MonotonicSingleProbitBayesOpt": {"restarts": 10, "samps": 1000},
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
        "modelbridge_cls": "MonotonicSingleProbitBayesOpt",
        "init_strat_cls": "SobolStrategy",
        "opt_strat_cls": "EpsilonGreedyModelWrapperStrategy",
        "model": "MonotonicGPLSETS",
        "parnames": "[context,intensity]",
    },
    "MonotonicMCLSE": {"target": 0.75, "beta": 3.98,},
    "MonotonicGPLSETS": {
        "inducing_size": 100,
        "mean_covar_factory": [
            "conditioning_mean_covar_factory",
            "monotonic_mean_covar_factory",
        ],
        "monotonic_idxs": ["[1]", "[]"],
        "uniform_idxs": "[]",
    },
    "conditioning_mean_covar_factory": {
        "fval_upper": "4",
        "fval_lower": "-4",
        "intensity_dim": "1",
        "cond_noise": "0",
        "num_pinned_induc": "10",
        "base_factory": "monotonic_mean_covar_factory",
    },
    "MonotonicSingleProbitBayesOpt": {"restarts": 10, "samps": 1000},
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
        # hack: on discrimination
        # test function instead of phi(f(0))\approx 0,
        # we need phi(f(0)) = 0.5 or f(0)=0
        # and on detection phi(0)=0 is too stringent
        if (
            name == "novel_discrimination"
            and "conditioning_mean_covar_factory" in full_config.keys()
        ):
            full_config["conditioning_mean_covar_factory"]["fval_lower"] = "0"
        elif (
            name == "novel_detection"
            and "conditioning_mean_covar_factory" in full_config.keys()
        ):
            full_config["conditioning_mean_covar_factory"]["fval_lower"] = "-3"
        benches.append(
            bench_cls(
                problem=problem,
                logger=logger,
                configs=full_config,
                global_seed=global_seed,
                n_reps=n_reps,
            )
        )
    return combine_benchmarks(*benches)


all_benchmarks = [
    make_bench(testfun, combo_logger, name, all_bench_configs, **bounds)
    for (testfun, bounds, name) in zip(all_testfuns, all_bounds, all_names)
]

from itertools import product
from copy import deepcopy
from tqdm import tqdm
import pathos
import functools
import itertools

mp = pathos.helpers.mp
nproc = 94

augmented_args = []
for bench in all_benchmarks:
    args = [
        (config_dict, seed, rep)
        for seed, (config_dict, rep) in enumerate(
            product(bench.combinations, range(bench.n_reps))
        )
    ]
    augmented_args.extend(itertools.product([bench], args))


manager = mp.Manager()
cache = manager.dict()


def memoize(func):
    @functools.wraps(func)
    def memoized(*args):
        result = None
        key = str(args)
        if key in cache:
            result = cache[key]
        else:
            result = func(*args)
            cache[key] = result
        return result

    return memoized


@memoize
def runexp(bench, args):
    config_dict, seed, rep = args
    # copy things that we mutate
    local_config = deepcopy(config_dict)
    local_logger = BenchmarkLogger(log_every=log_every)
    # _ = bench.run_experiment(local_config, local_logger, seed, rep)
    try:
        _ = bench.run_experiment(local_config, local_logger, seed, rep)
    except Exception as e:  # don't kill the pool
        print(
            f"Exception [{e}] raised by config_dict: {config_dict}\n with seed {seed} and rep {rep}"
        )
    return local_logger


import random
import time

random.shuffle(augmented_args)  # so we get results across the whole space gradually

start = time.time()
with mp.Pool(processes=nproc) as p:
    res = p.starmap(runexp, augmented_args)
end = time.time()
dur = end - start


outs = res
tmplogger = BenchmarkLogger(log_every=1)
for l in outs:
    tmplogger._log.extend(l._log)

# out_pd = tmplogger.pandas()
# out_pd.to_csv(f"mise_seed_{global_seed}.csv")
out_pd = tmplogger.pandas()
out_pd.to_csv(f"bigbench_pinned_partial_seed_{global_seed}.csv")
# out_pd.to_csv(f"smallbench_pinned_final_seed_{global_seed}.csv")


# out_pd = tmplogger.pandas()
# out_pd.to_csv(f"bigbench_newpriors_final_seed_{global_seed}.csv")
