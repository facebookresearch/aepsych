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
import numpy as np
from aepsych.server import AEPsychServer
from scipy.stats import norm

# force torch to 1 thread too just in case
torch.set_num_interop_threads(1)
torch.set_num_threads(1)

from aepsych.benchmark import run_benchmarks_with_checkpoints
from aepsych.benchmark import Problem

from aepsych.benchmark.test_functions import modified_hartmann6
from aepsych_prerelease.benchmark.test_functions import make_hairtie_parameterized
from aepsych.utils import make_scaled_sobol


import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.ERROR)

bench_name = "benchmark_pairwise_kernel"
out_path = f"./output/{bench_name}"

chunks = 10 # number of saved chuncks
reps_per_chunk = 20 # experiments per chunck
nproc = 62 # number of cores
global_seed = 1000 # All other seeds are created relative to this seed
log_every = 5 # Record metrics after this many new points
checkpoint_every = 600 # Sleep timer (sec) to check if resutls are ready

class PairwiseKernel_Problem(Problem):
    """Reformats the Problem class for the pairwise kernel data
    """
    def __init__(
        self,
        lb: torch.tensor = None,
        ub: torch.tensor = None,
    ) -> None:

        if lb is not None and ub is not None:
            self._bounds = torch.tensor(np.c_[lb, ub]).T

    @property
    def eval_grid(self):
        return make_scaled_sobol(
            lb=self.lb.tile((2,)),
            ub=self.ub.tile((2,)),
            size=self.n_eval_points
        )

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

    @property
    def f_true(self) -> np.ndarray:
        """Evaluate true test function over evaluation grid.

        Returns:
            torch.Tensor: Values of true test function over evaluation grid.
        """
        _eval_grid = self.eval_grid
        _shape = _eval_grid.shape
        _eval_grid = _eval_grid.reshape(-1, _shape[0], 2)
        return torch.tensor(self.f(_eval_grid)).detach().numpy()

    def f_pairwise(self, f, x, noise_scale=1):
        return norm.cdf((f(x[..., 1]) - f(x[..., 0])) / (noise_scale * np.sqrt(2)))

# Build a problem class for wach test function
class Hartmann6_Problem(PairwiseKernel_Problem):
    @property
    def name(self) -> str:
        return "PWKernel_hartmann_6D"

    def f(self, x):
        return self.f_pairwise(self.f_hartmann6, x, noise_scale=1)

    def f_hartmann6(self, x):
        return 3 * modified_hartmann6(x) - 2

class Hairtie_Problem(PairwiseKernel_Problem):
    @property
    def name(self) -> str:
        return "PWKernel_hairtie_2D"

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
        "model": "GPClassificationModel",
    },
    "PairwiseMCPosteriorVariance": {
        "objective": "ProbitObjective",
    },
    "GPClassificationModel": {
        "inducing_size": 100,
        "mean_covar_factory": "pairwise_kernel_factory",
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
