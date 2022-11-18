#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, List, Optional, Union

import gpytorch
import numpy as np
import torch
from aepsych.config import Config
from aepsych.factory.factory import default_mean_covar_factory
from aepsych.models.gp_classification import GPClassificationModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.likelihoods import Likelihood
from statsmodels.stats.moment_helpers import corr2cov, cov2corr


class MonotonicProjectionGP(GPClassificationModel):
    """A monotonic GP based on posterior projection

    NOTE: This model does not currently support backprop and so cannot be used
    with gradient optimization for active learning.

    This model produces predictions that are monotonic in any number of
    specified monotonic dimensions. It follows the intuition of the paper

    Lin L, Dunson DB (2014) Bayesian monotone regression using Gaussian process
    projection, Biometrika 101(2): 303-317.

    but makes significant departures by using heuristics for a lot of what is
    done in a more principled way in the paper. The reason for the move to
    heuristics is to improve scaling, especially with multiple monotonic
    dimensions.

    The method in the paper applies PAVA projection at the sample level,
    which requires a significant amount of costly GP posterior sampling. The
    approach taken here applies rolling-max projection to quantiles of the
    distribution, and so requires only marginal posterior evaluation. There is
    also a significant departure in the way multiple monotonic dimensions are
    handled, since in the paper computation scales exponentially with the
    number of monotonic dimensions and the heuristic approach taken here scales
    linearly in the number of dimensions.

    The cost of these changes is that the convergence guarantees proven in the
    paper no longer hold. The method implemented here is a heuristic, and it
    may be useful in some problems.

    The principle behind the method given here is that sample-level
    monotonicity implies monotonicity in the quantiles. We enforce monotonicity
    in several quantiles, and use that as an approximation for the true
    projected posterior distribution.

    The approach here also supports specifying a minimum value of f. That
    minimum will be enforced on mu, but not necessarily on the lower bound
    of the projected posterior since we keep the projected posterior normal.
    The min f value will also be enforced on samples drawn from the model,
    while monotonicity will not be enforced at the sample level.

    The procedure for computing the monotonic projected posterior at x is:
    1. Separately for each monotonic dimension, create a grid of s points that
    differ only in that dimension, and sweep from the lower bound up to x.
    2. Evaluate the marginal distribution, mu and sigma, on the full set of
    points (x and the s grid points for each monotonic dimension).
    3. Compute the mu +/- 2 * sigma quantiles.
    4. Enforce monotonicity in the quantiles by taking mu_proj as the maximum
    mu across the set, and lb_proj as the maximum of mu - 2 * sigma across the
    set. ub_proj is left as mu(x) + 2 * sigma(x), but is clamped to mu_proj in
    case that project put it above the original ub.
    5. Clamp mu and lb to the minimum value for f, if one was set.
    6. Construct a new normal posterior given the projected quantiles by taking
    mu_proj as the mean, and (ub - lb) / 4 as the standard deviation. Adjust
    the covariance matrix to account for the change in the marginal variances.

    The process above requires only marginal posterior evaluation on the grid
    of points used for the posterior projection, and the size of that grid
    scales linearly with the number of monotonic dimensions, not exponentially.

    The args here are the same as for GPClassificationModel with the addition
    of:

    Args:
        monotonic_dims: A list of the dimensions on which monotonicity should
            be enforced.
        monotonic_grid_size: The size of the grid, s, in 1. above.
        min_f_val: If provided, maintains this minimum in the projection in 5.
    """

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        monotonic_dims: List[int],
        monotonic_grid_size: int = 20,
        min_f_val: Optional[float] = None,
        dim: Optional[int] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        inducing_size: int = 100,
        max_fit_time: Optional[float] = None,
        inducing_point_method: str = "auto",
    ):
        assert len(monotonic_dims) > 0
        self.monotonic_dims = monotonic_dims
        self.mon_grid_size = monotonic_grid_size
        self.min_f_val = min_f_val
        super().__init__(
            lb=lb,
            ub=ub,
            dim=dim,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            inducing_size=inducing_size,
            max_fit_time=max_fit_time,
            inducing_point_method=inducing_point_method,
        )

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise: Union[bool, torch.Tensor] = False,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        # Augment X with monotonicity grid points, for each monotonic dim
        n, d = X.shape  # Require no batch dimensions
        m = len(self.monotonic_dims)
        s = self.mon_grid_size
        X_aug = X.repeat(s * m + 1, 1, 1)
        for i, dim in enumerate(self.monotonic_dims):
            # using numpy because torch doesn't support vectorized linspace,
            # pytorch/issues/61292
            grid: Union[np.ndarray, torch.Tensor] = np.linspace(
                self.lb[dim],
                X[:, dim].numpy(),
                s + 1,
            )  # (s+1 x n)
            grid = torch.tensor(grid[:-1, :], dtype=X.dtype)  # Drop x; (s x n)
            X_aug[(1 + i * s) : (1 + (i + 1) * s), :, dim] = grid
        # X_aug[0, :, :] is X, and then subsequent indices are points in the grids
        # Predict marginal distributions on X_aug
        with torch.no_grad():
            post_aug = super().posterior(X=X_aug)
        mu_aug = post_aug.mean.squeeze()  # (m*s+1 x n)
        var_aug = post_aug.variance.squeeze()  # (m*s+1 x n)
        mu_proj = mu_aug.max(dim=0).values
        lb_proj = (mu_aug - 2 * torch.sqrt(var_aug)).max(dim=0).values
        if self.min_f_val is not None:
            mu_proj = mu_proj.clamp(min=self.min_f_val)
            lb_proj = lb_proj.clamp(min=self.min_f_val)
        ub_proj = (mu_aug[0, :] + 2 * torch.sqrt(var_aug[0, :])).clamp(min=mu_proj)
        sigma_proj = ((ub_proj - lb_proj) / 4).clamp(min=1e-4)
        # Adjust the whole covariance matrix to accomadate the projected marginals
        with torch.no_grad():
            post = super().posterior(X=X)
            R = cov2corr(post.distribution.covariance_matrix.squeeze().numpy())
            S_proj = torch.tensor(corr2cov(R, sigma_proj.numpy()), dtype=X.dtype)
        mvn_proj = gpytorch.distributions.MultivariateNormal(
            mu_proj.unsqueeze(0),
            S_proj.unsqueeze(0),
        )
        return GPyTorchPosterior(mvn_proj)

    def sample(
        self, x: Union[torch.Tensor, np.ndarray], num_samples: int
    ) -> torch.Tensor:
        samps = super().sample(x=x, num_samples=num_samples)
        if self.min_f_val is not None:
            samps = samps.clamp(min=self.min_f_val)
        return samps

    @classmethod
    def from_config(cls, config: Config) -> MonotonicProjectionGP:
        """Alternate constructor for MonotonicProjectionGP model.

        This is used when we recursively build a full sampling strategy
        from a configuration. TODO: document how this works in some tutorial.

        Args:
            config (Config): A configuration containing keys/values matching this class

        Returns:
            MonotonicProjectionGP: Configured class instance.
        """

        classname = cls.__name__
        inducing_size = config.getint(classname, "inducing_size", fallback=10)

        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = config.getint(classname, "dim", fallback=None)

        mean_covar_factory = config.getobj(
            classname, "mean_covar_factory", fallback=default_mean_covar_factory
        )

        mean, covar = mean_covar_factory(config)
        max_fit_time = config.getfloat(classname, "max_fit_time", fallback=None)

        inducing_point_method = config.get(
            classname, "inducing_point_method", fallback="auto"
        )

        likelihood_cls = config.getobj(classname, "likelihood", fallback=None)

        if likelihood_cls is not None:
            if hasattr(likelihood_cls, "from_config"):
                likelihood = likelihood_cls.from_config(config)
            else:
                likelihood = likelihood_cls()
        else:
            likelihood = None  # fall back to __init__ default

        monotonic_dims: List[int] = config.getlist(
            classname, "monotonic_dims", fallback=[-1]
        )
        monotonic_grid_size = config.getint(
            classname, "monotonic_grid_size", fallback=20
        )
        min_f_val = config.getfloat(classname, "min_f_val", fallback=None)

        return cls(
            lb=lb,
            ub=ub,
            dim=dim,
            inducing_size=inducing_size,
            mean_module=mean,
            covar_module=covar,
            max_fit_time=max_fit_time,
            inducing_point_method=inducing_point_method,
            likelihood=likelihood,
            monotonic_dims=monotonic_dims,
            monotonic_grid_size=monotonic_grid_size,
            min_f_val=min_f_val,
        )
