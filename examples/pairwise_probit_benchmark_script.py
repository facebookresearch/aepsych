import dill
dill.settings['recurse'] = True

import os
import logging

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

import numpy as np
from typing import Dict, Union

import aepsych
from aepsych.strategy import SequentialStrategy, Strategy
from aepsych.server import AEPsychServer
from aepsych.benchmark import run_benchmarks_with_checkpoints
from aepsych.benchmark import Problem
from aepsych.benchmark.test_functions import modified_hartmann6
from aepsych_prerelease.benchmark.test_functions import make_hairtie_parameterized
import aepsych.utils_logging as utils_logging
from scipy.stats import norm, pearsonr

logger = utils_logging.getLogger(logging.ERROR)

bench_name = "benchmark_pairwise_probit"
out_path = f"./output/{bench_name}"

chunks = 10 # number of saved chuncks
reps_per_chunk = 20 # experiments per chunck
nproc = 62 # number of cores
global_seed = 1000 # All other seeds are created relative to this seed
log_every = 5 # Record metrics after this many new points
checkpoint_every = 600 # Sleep timer (sec) to check if resutls are ready

class PairwiseProbit_Problem(Problem):
    """Reformats the Problem class for the pairwise probit data
    """
    def __init__(
        self,
        lb: torch.tensor = None,
        ub: torch.tensor = None,
    ) -> None:

        if lb is not None and ub is not None:
            self._bounds = torch.tensor(np.c_[lb, ub]).T

    @property
    def bounds(self):
        return self._bounds

    def p(self, x: np.ndarray) -> np.ndarray:
        """Evaluate response probability from test function.

        Args:
            x (np.ndarray): Points at which to evaluate.

        Returns:
            np.ndarray: Response probability at queries points.
        """
        return norm.cdf(self.f(x[0]))

    def p_hat(self, model: aepsych.models.base.ModelProtocol) -> torch.Tensor:
        """Generate mean predictions from the model over the evaluation grid.

        Args:
            model (aepsych.models.base.ModelProtocol): Model to evaluate.

        Returns:
            torch.Tensor: Posterior mean from underlying model over the evaluation grid.
        """
        p_hat, _ = model.predict(self.eval_grid)
        return torch.tensor(norm.cdf(p_hat.detach().numpy()))

    @property
    def f_true(self) -> np.ndarray:
        """Evaluate true test function over evaluation grid.

        Returns:
            torch.Tensor: Values of true test function over evaluation grid.
        """
        _eval_grid = np.c_[self.eval_grid, self.eval_grid]
        _shape = _eval_grid.shape
        _eval_grid = _eval_grid.reshape(-1, _shape[0], 2)
        return torch.tensor(self.f(_eval_grid)).detach().numpy()

    def f_pairwise(self, f, x, noise_scale=1):
        return norm.cdf((f(x[..., 1]) - f(x[..., 0])) / (noise_scale * np.sqrt(2)))

    def evaluate(
        self,
        strat: Union[Strategy, SequentialStrategy],
    ) -> Dict[str, float]:
        """Evaluate the strategy with respect to this problem.

        Extend this in subclasses to add additional metrics.
        Metrics include:
        - mae (mean absolute error), mae (mean absolute error), max_abs_err (max absolute error),
            pearson correlation. All of these are computed over the latent variable f and the
            outcome probability p, w.r.t. the posterior mean. Squared and absolute errors (miae, mise) are
            also computed in expectation over the posterior, by sampling.
        - Brier score, which measures how well-calibrated the outcome probability is, both at the posterior
            mean (plain brier) and in expectation over the posterior (expected_brier).

        Args:
            strat (aepsych.strategy.Strategy): Strategy to evaluate.

        Returns:
            Dict[str, float]: A dictionary containing metrics and their values.
        """
        # we just use model here but eval gets called on strat in case we need it in downstream evals
        # for example to separate out sobol vs opt trials
        model = strat.model
        assert model is not None, "Cannot evaluate strategy without a model!"

        # always eval f
        f_hat = self.f_hat(model).detach().numpy()
        p_hat = self.p_hat(model).detach().numpy()
        assert (
            self.f_true.shape == f_hat.shape
        ), f"self.f_true.shape=={self.f_true.shape} != f_hat.shape=={f_hat.shape}"

        mae_f = np.mean(np.abs(self.f_true - f_hat))
        mse_f = np.mean((self.f_true - f_hat) ** 2)
        max_abs_err_f = np.max(np.abs(self.f_true - f_hat))
        corr_f = pearsonr(self.f_true.flatten(), f_hat.flatten())[0]
        mae_p = np.mean(np.abs(self.p_true - p_hat))
        mse_p = np.mean((self.p_true - p_hat) ** 2)
        max_abs_err_p = np.max(np.abs(self.p_true - p_hat))
        corr_p = pearsonr(self.p_true.flatten(), p_hat.flatten())[0]
        brier = np.mean(2 * np.square(self.p_true - p_hat))

        # eval in samp-based expectation over posterior instead of just mean
        fsamps = model.sample(self.eval_grid, num_samples=1000).detach().numpy()
        try:
            psamps = (
                model.sample(self.eval_grid, num_samples=1000, probability_space=True)  # type: ignore
                .detach()
                .numpy()
            )
        except TypeError:  # vanilla models don't have proba_space samps, TODO maybe we should add them
            psamps = norm.cdf(fsamps)

        fsamps = fsamps[..., 0]
        psamps = fsamps[..., 0]

        ferrs = fsamps - self.f_true[None, :]
        miae_f = np.mean(np.abs(ferrs))
        mise_f = np.mean(ferrs**2)

        perrs = psamps - self.p_true[None, :]
        miae_p = np.mean(np.abs(perrs))
        mise_p = np.mean(perrs**2)

        expected_brier = (2 * np.square(self.p_true[None, :] - psamps)).mean()

        metrics = {
            "mean_abs_err_f": mae_f,
            "mean_integrated_abs_err_f": miae_f,
            "mean_square_err_f": mse_f,
            "mean_integrated_square_err_f": mise_f,
            "max_abs_err_f": max_abs_err_f,
            "pearson_corr_f": corr_f,
            "mean_abs_err_p": mae_p,
            "mean_integrated_abs_err_p": miae_p,
            "mean_square_err_p": mse_p,
            "mean_integrated_square_err_p": mise_p,
            "max_abs_err_p": max_abs_err_p,
            "pearson_corr_p": corr_p,
            "brier": brier,
            "expected_brier": expected_brier,
        }

        return metrics

# Build a problem class for wach test function
class Hartmann6_Problem(PairwiseProbit_Problem):
    @property
    def name(self) -> str:
        return "PWProbit_hartmann_6D"

    def f(self, x):
        return self.f_pairwise(self.f_hartmann6, x, noise_scale=1)

    def f_hartmann6(self, x):
        return 3 * modified_hartmann6(x) - 2.0

class Hairtie_Problem(PairwiseProbit_Problem):
    @property
    def name(self) -> str:
        return "PWProbit_hairtie_2D"

    def f(self, x):
        return self.f_pairwise(self.f_hairtie, x, noise_scale=1)

    def f_hairtie(self, x):
        ht = make_hairtie_parameterized(
            vscale=1.0, vshift=0.05, variance=0.1,
            n_interval=1, asym=0.5, phase=0, period=1
        )
        return ht(x.T)


# Initilize problem class and make config
problem1 = Hartmann6_Problem(
    lb=torch.zeros(6, dtype=torch.double),
    ub=torch.ones(6, dtype=torch.double),
)
problem2 = Hairtie_Problem(
    lb=torch.tensor([-1, -1]),
    ub=torch.tensor([1, 1]),
)
problems = [problem1, problem2]
problems = [problem1]
bench_config = {
    "common": {
        "stimuli_per_trial": 2,
        "outcome_types": ["binary"],
        "strategy_names": "[init_strat, opt_strat]",
    },
    "init_strat": {
        "min_asks": 10,
        "generator": "SobolGenerator",
    },
    "opt_strat": {
        "min_asks": 1000,
        "refit_every": 5,
        "generator": "SobolGenerator",
        "model": "PairwiseProbitModel",
    },
    "PairwiseMCPosteriorVariance": {
        "objective": "ProbitObjective",
    },
    "PairwiseProbitModel": {
        "inducing_size": 100,
        "mean_covar_factory": "default_mean_covar_factory",
    },
}


if __name__ == "__main__":
    run_benchmarks_with_checkpoints(
        out_path=out_path,
        benchmark_name=bench_name,
        problems=problems,
        configs=bench_config,
        global_seed=global_seed,
        n_chunks=chunks,
        n_reps_per_chunk=reps_per_chunk,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        n_proc=nproc,
        serial_debug=False,
    )
