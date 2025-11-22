#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from aepsych.benchmark import (
    Benchmark,
    BenchmarkLogger,
    combine_benchmarks,
    LSEProblem,
    Problem,
)
from aepsych.benchmark.test_functions import (
    make_songetal_testfun,
    novel_detection_testfun,
    novel_discrimination_testfun,
)
from aepsych.config import Config
from aepsych.plotting import plot_strat
from aepsych.strategy import SequentialStrategy
from scipy.stats import norm

global_seed = 3
refit_every = 1
figdir = "./figs/"


def plot_audiometric_lse_grids(
    sobol_trials, opt_trials, phenotype="Metabolic+Sensory", beta=2
):
    """
    Generates Fig. 8
    """

    logger = BenchmarkLogger(log_every=5)
    bench_rbf = {
        "common": {"pairwise": False, "target": 0.75},
        "experiment": {
            "acqf": "MonotonicMCLSE",
            "modelbridge_cls": "MonotonicSingleProbitModelbridge",
            "init_strat_cls": "SobolStrategy",
            "opt_strat_cls": "ModelWrapperStrategy",
            "model": "MonotonicRejectionGP",
            "parnames": "[context,intensity]",
        },
        "MonotonicMCLSE": {
            "target": 0.75,
            "beta": 3.84,
        },
        "MonotonicRejectionGP": {
            "inducing_size": 100,
            "mean_covar_factory": [
                "monotonic_mean_covar_factory",
            ],
            "monotonic_idxs": ["[1]", "[]"],
            "uniform_idxs": "[]",
        },
        "MonotonicSingleProbitModelbridge": {"restarts": 10, "samps": 1000},
        "SobolStrategy": {
            "n_trials": [sobol_trials],
        },
        "ModelWrapperStrategy": {
            "n_trials": [opt_trials],
            "refit_every": [refit_every],
        },
    }
    bench_song = {
        "common": {"pairwise": False, "target": 0.75},
        "experiment": {
            "acqf": "BernoulliMCMutualInformation",
            "modelbridge_cls": "SingleProbitModelbridgeWithSongHeuristic",
            "init_strat_cls": "SobolStrategy",
            "opt_strat_cls": "ModelWrapperStrategy",
            "model": "GPClassificationModel",
            "parnames": "[context,intensity]",
        },
        "GPClassificationModel": {
            "inducing_size": 100,
            "dim": 2,
            "mean_covar_factory": [
                "song_mean_covar_factory",
            ],
        },
        "SingleProbitModelbridgeWithSongHeuristic": {"restarts": 10, "samps": 1000},
        "SobolStrategy": {
            "n_trials": [sobol_trials],
        },
        "ModelWrapperStrategy": {
            "n_trials": [opt_trials],
            "refit_every": [refit_every],
        },
    }

    all_bench_configs = [bench_rbf, bench_song]

    testfun = make_songetal_testfun(phenotype=phenotype, beta=beta)

    class AudiometricProblem(LSEProblem, Problem):
        def f(self, x):
            return testfun(x)

    lb = [-3, -20]
    ub = [4, 120]
    benches = []
    problem = AudiometricProblem(lb, ub)
    for config in all_bench_configs:
        full_config = copy(config)
        full_config["common"]["lb"] = str(lb)
        full_config["common"]["ub"] = str(ub)
        benches.append(
            Benchmark(
                problem=problem,
                logger=logger,
                configs=full_config,
                global_seed=global_seed,
                n_reps=1,
            )
        )
    combo_bench = combine_benchmarks(*benches)
    strats = []

    for config in combo_bench.combinations:
        strat = combo_bench.run_experiment(config, logger, seed=global_seed, rep=0)
        strats.append(strat)

    titles = [
        "Monotonic RBF Model, LSE (ours)",
        "Nonmonotonic RBF Model, LSE (ours)",
        "Linear-Additive Model, BALD",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.5))
    plotting_axes = [axes[1, 0], axes[0, 1], axes[0, 0]]
    fig.delaxes(axes[1, 1])
    _ = [
        plot_strat(
            strat=strat_,
            title=title_,
            ax=ax_,
            true_testfun=testfun,
            xlabel="Frequency (kHz)",
            ylabel="Intensity (dB HL)",
            flipx=True,
            logx=True,
            show=False,
            include_legend=False,
            include_colorbar=False,
        )
        for ax_, strat_, title_ in zip(plotting_axes, strats, titles)
    ]
    fig.tight_layout()
    handles, labels = axes[1, 0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.8, 0.2))
    cbr = fig.colorbar(axes[1, 0].images[0], ax=plotting_axes)
    cbr.set_label("Probability of Detection")

    return fig


def plot_novel_lse_grids(sobol_trials, opt_trials, funtype="detection"):
    """
    Generates Fig. TBA
    """

    logger = BenchmarkLogger(log_every=opt_trials)  # we only care about final perf
    bench_rbf = {
        "common": {"pairwise": False, "target": 0.75},
        "experiment": {
            "acqf": "MonotonicMCLSE",
            "modelbridge_cls": "MonotonicSingleProbitModelbridge",
            "init_strat_cls": "SobolStrategy",
            "opt_strat_cls": "ModelWrapperStrategy",
            "model": "MonotonicRejectionGP",
            "parnames": "[context,intensity]",
        },
        "MonotonicMCLSE": {
            "target": 0.75,
            "beta": 3.84,
        },
        "MonotonicRejectionGP": {
            "inducing_size": 100,
            "mean_covar_factory": [
                "monotonic_mean_covar_factory",
            ],
            "monotonic_idxs": ["[1]", "[]"],
            "uniform_idxs": "[]",
        },
        "MonotonicSingleProbitModelbridge": {"restarts": 10, "samps": 1000},
        "SobolStrategy": {
            "n_trials": [sobol_trials],
        },
        "ModelWrapperStrategy": {
            "n_trials": [opt_trials],
            "refit_every": [refit_every],
        },
    }
    bench_song = {
        "common": {"pairwise": False, "target": 0.75},
        "experiment": {
            "acqf": "BernoulliMCMutualInformation",
            "modelbridge_cls": "SingleProbitModelbridgeWithSongHeuristic",
            "init_strat_cls": "SobolStrategy",
            "opt_strat_cls": "ModelWrapperStrategy",
            "model": "GPClassificationModel",
            "parnames": "[context,intensity]",
        },
        "GPClassificationModel": {
            "inducing_size": 100,
            "dim": 2,
            "mean_covar_factory": [
                "song_mean_covar_factory",
            ],
        },
        "SingleProbitModelbridgeWithSongHeuristic": {"restarts": 10, "samps": 1000},
        "SobolStrategy": {
            "n_trials": [sobol_trials],
        },
        "ModelWrapperStrategy": {
            "n_trials": [opt_trials],
            "refit_every": [refit_every],
        },
    }
    all_bench_configs = [bench_rbf, bench_song]

    if funtype == "detection":
        testfun = novel_detection_testfun
        yes_label = "Detected trial"
        no_label = "Nondetected trial"
    elif funtype == "discrimination":
        testfun = novel_discrimination_testfun
        yes_label = "Correct trial"
        no_label = "Incorrect trial"
    else:
        raise RuntimeError("unknown testfun")

    class NovelProblem(LSEProblem, Problem):
        def f(self, x):
            return testfun(x)

    lb = [-1, -1]
    ub = [1, 1]
    benches = []
    problem = NovelProblem(lb, ub, gridsize=50)
    for config in all_bench_configs:
        full_config = copy(config)
        full_config["common"]["lb"] = str(lb)
        full_config["common"]["ub"] = str(ub)
        benches.append(
            Benchmark(
                problem=problem,
                logger=logger,
                configs=full_config,
                global_seed=global_seed,
                n_reps=1,
            )
        )
    combo_bench = combine_benchmarks(*benches)
    strats = []

    for config in combo_bench.combinations:
        strat = combo_bench.run_experiment(config, logger, seed=global_seed, rep=0)
        strats.append(strat)

    titles = [
        "Monotonic RBF Model, LSE (ours)",
        "Nonmonotonic RBF Model, LSE (ours)",
        "Linear-Additive Model, BALD",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.5))
    plotting_axes = [axes[1, 0], axes[0, 1], axes[0, 0]]
    fig.delaxes(axes[1, 1])
    _ = [
        plot_strat(
            strat=strat_,
            title=title_,
            ax=ax_,
            true_testfun=testfun,
            yes_label=yes_label,
            no_label=no_label,
            show=False,
            include_legend=False,
            include_colorbar=False,
        )
        for ax_, strat_, title_ in zip(plotting_axes, strats, titles)
    ]
    fig.tight_layout()
    handles, labels = axes[1, 0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.8, 0.2))
    cbr = fig.colorbar(axes[1, 0].images[0], ax=plotting_axes)
    cbr.set_label("Probability of Detection")

    return fig


def plot_acquisition_examples(sobol_trials, opt_trials, target_level=0.75):
    ### Same model, different acqf figure ####

    configs = {
        "common": {
            "pairwise": False,
            "target": target_level,
            "lb": "[-3]",
            "ub": "[3]",
        },
        "experiment": {
            "acqf": [
                "MonotonicMCPosteriorVariance",
                "MonotonicBernoulliMCMutualInformation",
                "MonotonicMCLSE",
            ],
            "modelbridge_cls": "MonotonicSingleProbitModelbridge",
            "init_strat_cls": "SobolStrategy",
            "opt_strat_cls": "ModelWrapperStrategy",
            "model": "MonotonicRejectionGP",
            "parnames": "[intensity]",
        },
        "MonotonicMCLSE": {
            "target": target_level,
            "beta": 3.84,
        },
        "MonotonicRejectionGP": {
            "inducing_size": 100,
            "mean_covar_factory": "monotonic_mean_covar_factory",
            "monotonic_idxs": "[0]",
            "uniform_idxs": "[]",
        },
        "MonotonicSingleProbitModelbridge": {"restarts": 10, "samps": 1000},
        "SobolStrategy": {"n_trials": sobol_trials},
        "ModelWrapperStrategy": {
            "n_trials": opt_trials,
            "refit_every": refit_every,
        },
    }

    def true_testfun(x):
        return norm.cdf(3 * x)

    class SimpleLinearProblem(Problem):
        def f(self, x):
            return norm.ppf(true_testfun(x))

    lb = [-3]
    ub = [3]

    logger = BenchmarkLogger()
    problem = SimpleLinearProblem(lb, ub)
    bench = Benchmark(
        problem=problem,
        logger=logger,
        configs=configs,
        global_seed=global_seed,
        n_reps=1,
    )

    # sobol_trials
    # now run each for just init trials, taking care to reseed each time
    strats = []
    for c in bench.combinations:
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        s = SequentialStrategy.from_config(Config(config_dict=c))
        for _ in range(sobol_trials):
            next_x = s.gen()
            s.add_data(next_x, [problem.sample_y(next_x)])
        strats.append(s)

    # get first gen from all 3
    first_gens = [s.gen() for s in strats]

    fig, ax = plt.subplots(2, 2)
    plot_strat(
        strat=strats[0],
        title=f"First active trial\n (after {sobol_trials} Sobol trials)",
        ax=ax[0, 0],
        true_testfun=true_testfun,
        target_level=target_level,
        show=False,
        include_legend=False,
    )
    samps = [
        norm.cdf(s.sample(torch.Tensor(g), num_samples=10000))
        for s, g in zip(strats, first_gens)
    ]
    predictions = [np.mean(s) for s in samps]
    names = ["First BALV sample", "First BALD sample", "First LSE sample"]
    markers = ["s", "*", "^"]
    for i in range(3):
        ax[0, 0].scatter(
            first_gens[i][0][0],
            predictions[i],
            label=names[i],
            marker=markers[i],
            color="black",
        )

    # now run them all for the full duration
    for s in strats:
        for _tr in range(opt_trials):
            next_x = s.gen()
            s.add_data(next_x, [problem.sample_y(next_x)])

    plotting_axes = [ax[0, 1], ax[1, 0], ax[1, 1]]

    titles = [
        f"Monotonic RBF Model,\n BALV, after {sobol_trials + opt_trials} total trials",
        f"Monotonic RBF Model,\n BALD, after {sobol_trials + opt_trials} total trials",
        f"Monotonic RBF Model,\n LSE (ours) after {sobol_trials + opt_trials} total trials",
    ]

    _ = [
        plot_strat(
            strat=s,
            title=t,
            ax=a,
            true_testfun=true_testfun,
            target_level=target_level,
            show=False,
            include_legend=False,
        )
        for a, s, t in zip(plotting_axes, strats, titles)
    ]
    fig.tight_layout()
    handles, labels = ax[0, 0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.5, 0.25))
    # return legend so savefig works correctly
    return fig, lgd


if __name__ == "__main__":
    audio_lse_grids_fig = plot_audiometric_lse_grids(sobol_trials=5, opt_trials=45)
    audio_lse_grids_fig.savefig(fname=figdir + "audio_lse_grids_fig.pdf", dpi=200)
    novel_detection_lse_grids_fig = plot_novel_lse_grids(
        sobol_trials=5, opt_trials=45, funtype="detection"
    )
    novel_detection_lse_grids_fig.savefig(
        fname=figdir + "detection_lse_grids_fig.pdf", dpi=200
    )

    # this is extra hard, run more trials
    novel_discrimination_lse_grids_fig = plot_novel_lse_grids(
        sobol_trials=5, opt_trials=95, funtype="discrimination"
    )
    novel_discrimination_lse_grids_fig.savefig(
        fname=figdir + "discrimination_lse_grids_fig.pdf", dpi=200
    )

    same_model_different_acq_fig, lgd = plot_acquisition_examples(
        sobol_trials=5, opt_trials=15
    )

    same_model_different_acq_fig.savefig(
        fname=figdir + "same_model_different_acq.pdf",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        dpi=200,
    )
