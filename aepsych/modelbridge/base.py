import logging
from inspect import signature

import aepsych.utils_logging as utils_logging
import gpytorch
import numpy as np
import torch
from aepsych.utils import promote_0d, _dim_grid, get_jnd_multid
from botorch.acquisition import (
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
    NoisyExpectedImprovement,
)
from scipy.stats import norm

# this is pretty aggressive jitter setting but should protect us from
# crashes which are a bigger concern in data collection.
# we refit with stan afterwards anyway.
gpytorch.settings.cholesky_jitter._global_float_value = 1e-3
gpytorch.settings.cholesky_jitter._global_double_value = 1e-3
gpytorch.settings.tridiagonal_jitter._global_value = 1e-3

logger = utils_logging.getLogger(logging.DEBUG)


def _prune_extra_acqf_args(acqf, extra_acqf_args):
    # prune extra args needed, ignore the rest
    # (this helps with API consistency)
    acqf_args_expected = signature(acqf).parameters.keys()
    return {k: v for k, v in extra_acqf_args.items() if k in acqf_args_expected}


def _prune_extra_acqf_args(acqf, extra_acqf_args):
    # prune extra args needed, ignore the rest
    # (this helps with API consistency)
    acqf_args_expected = signature(acqf).parameters.keys()
    return {k: v for k, v in extra_acqf_args.items() if k in acqf_args_expected}


class ModelBridge(object):
    """Base class for objects combining an interpolator/model, acquisition, and data
    Loosely inspired by https://ax.dev/api/modelbridge.html#module-ax.modelbridge.base
    but definitely not compatible with it.
    """

    baseline_requiring_acqfs = [qNoisyExpectedImprovement, NoisyExpectedImprovement]

    def __init__(self, lb, ub, dim=1, acqf=None, extra_acqf_args=None):
        self.dim = dim

        self.lb = torch.Tensor(promote_0d(lb)).float()
        self.ub = torch.Tensor(promote_0d(ub)).float()
        if extra_acqf_args is None:
            extra_acqf_args = {}

        if acqf is None:
            self.acqf = qUpperConfidenceBound
            extra_acqf_args["beta"] = 1.96
        else:
            self.acqf = acqf

        self.extra_acqf_args = _prune_extra_acqf_args(self.acqf, extra_acqf_args)
        self.objective = extra_acqf_args.get("objective", None)
        self.target = extra_acqf_args.get("target", 0.75)

    def gen(self):
        raise NotImplementedError("Implement me in subclasses!")

    def predict(self):
        raise NotImplementedError("Implement me in subclasses!")

    def update(self, *args, **kwargs):
        logger.info(
            "Calling update on a model without specialized "
            + "update implementation, defaulting to regular fit"
        )
        self.fit(*args, **kwargs)

    def sample(self, x, num_samples):
        """Override this as needed."""
        return self.model.posterior(x).sample(torch.Size([num_samples]))

    def _get_acquisition_fn(self):
        if self.acqf in self.baseline_requiring_acqfs:
            train_x = self.model.train_inputs[0]
            return self.acqf(
                model=self.model, X_baseline=train_x, **self.extra_acqf_args
            )
        else:
            return self.acqf(model=self.model, **self.extra_acqf_args)

    def best(self):
        from aepsych.plotting import lse_acqfs

        if self.acqf in lse_acqfs:
            return self._get_contour()
        else:
            pass

    def _get_contour(self, gridsize=30):

        from aepsych.utils import get_lse_contour

        if self.dim == 2:

            grid_search = _dim_grid(self, gridsize=gridsize)
            post_mean, _ = self.predict(torch.Tensor(grid_search))
            post_mean = norm.cdf(post_mean.reshape(gridsize, gridsize).detach().numpy())
            x1 = _dim_grid(lower=self.lb, upper=self.ub, dim=1, gridsize=gridsize)
            x2_hat = get_lse_contour(
                post_mean, x1, level=self.target, lb=x1.min(), ub=x1.max()
            )
            return x2_hat
        else:
            return None

    def get_jnd(
        self, grid=None, cred_level=None, intensity_dim=-1, confsamps=500, method="step"
    ):
        """Calculate the JND. Note that JND can have multiple plausible definitions
        outside of the linear case, so we provide options for how to compute it.
        For method="step", we report how far one needs to go over in stimulus
        space to move 1 unit up in latent space (this is a lot of people's
        conventional understanding of the JND).
        For method="taylor", we report the local derivative, which also maps to a
        1st-order Taylor expansion of the latent function. This is a formal
        generalization of JND as defined in Weber's law.
        Both definitions are equivalent for linear psychometric functions.
        """
        if grid is None:
            grid = _dim_grid(self)

        # this is super awkward, back into intensity dim grid assuming a square grid
        gridsize = int(np.power(grid.shape[0], 1 / grid.shape[1]))
        coords = np.linspace(self.lb[intensity_dim], self.ub[intensity_dim], gridsize)

        if cred_level is None:
            fmean, _ = self.predict(grid)
            fmean = fmean.detach().numpy().reshape(*[gridsize for i in range(self.dim)])

            if method == "taylor":
                return 1 / np.gradient(fmean, coords, axis=intensity_dim)
            elif method == "step":
                return np.clip(
                    get_jnd_multid(fmean, coords, mono_dim=intensity_dim), 0, np.inf
                )
        else:
            alpha = 1 - cred_level
            qlower = alpha / 2
            qupper = 1 - alpha / 2

            fsamps = self.sample(grid, confsamps)
            if method == "taylor":
                jnds = 1 / np.gradient(
                    fsamps.detach()
                    .numpy()
                    .reshape(confsamps, *[gridsize for i in range(self.dim)]),
                    coords,
                    axis=intensity_dim,
                )
            elif method == "step":
                samps = [
                    s.reshape((gridsize,) * self.dim) for s in fsamps.detach().numpy()
                ]
                jnds = np.stack(
                    [get_jnd_multid(s, coords, mono_dim=intensity_dim) for s in samps]
                )
            upper = np.clip(np.quantile(jnds, qupper, axis=0), 0, np.inf)
            lower = np.clip(np.quantile(jnds, qlower, axis=0), 0, np.inf)
            median = np.clip(np.quantile(jnds, 0.5, axis=0), 0, np.inf)
            return median, lower, upper

        raise RuntimeError(f"Unknown method {method}!")
