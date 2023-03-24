#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.models.model import Model
from botorch.models.utils.inducing_point_allocators import GreedyVarianceReduction
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from scipy.cluster.vq import kmeans2
from scipy.special import owens_t
from scipy.stats import norm
from torch.distributions import Normal


def compute_p_quantile(
    f_mean: torch.Tensor, f_std: torch.Tensor, alpha: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Compute quantile of p in probit model

    For f ~ N(mu_f, sigma_f^2) and p = Phi(f), computes the alpha quantile of p
    using the formula

    x = Phi(mu_f + Phi^-1(alpha) * sigma_f),

    which solves for x such that P(p <= x) = alpha.

    A 95% CI for p can be computed as
    p_l = compute_p_quantile(f_mean, f_std, 0.025)
    p_u = compute_p_quantile(f_mean, f_std, 0.975)
    """
    norm = torch.distributions.Normal(0, 1)
    alpha = torch.tensor(alpha, dtype=f_mean.dtype)
    return norm.cdf(f_mean + norm.icdf(alpha) * f_std)


def select_inducing_points(
    inducing_size: int,
    covar_module: Kernel = None,
    X: Optional[torch.Tensor] = None,
    bounds: Optional[Union[torch.Tensor, np.ndarray]] = None,
    method: str = "auto",
):
    with torch.no_grad():
        assert method in (
            "pivoted_chol",
            "kmeans++",
            "auto",
            "sobol",
        ), f"Inducing point method should be one of pivoted_chol, kmeans++, sobol, or auto; got {method}"

        if method == "sobol":
            assert bounds is not None, "Must pass bounds for sobol inducing points!"
            inducing_points = draw_sobol_samples(
                bounds=bounds, n=inducing_size, q=1
            ).squeeze()
            if len(inducing_points.shape) == 1:
                inducing_points = inducing_points.reshape(-1, 1)
            return inducing_points

        assert X is not None, "Must pass X for non-sobol inducing point selection!"
        # remove dupes from X, which is both wasteful for inducing points
        # and would break kmeans++
        unique_X = torch.unique(X, dim=0)
        if method == "auto":
            if unique_X.shape[0] <= inducing_size:
                return unique_X
            else:
                method = "kmeans++"

        if method == "pivoted_chol":
            inducing_point_allocator = GreedyVarianceReduction()
            inducing_points = inducing_point_allocator.allocate_inducing_points(
                inputs=X,
                covar_module=covar_module,
                num_inducing=inducing_size,
                input_batch_shape=torch.Size([]),
            )
        elif method == "kmeans++":
            # initialize using kmeans
            inducing_points = torch.tensor(
                kmeans2(unique_X.numpy(), inducing_size, minit="++")[0],
                dtype=X.dtype,
            )
        return inducing_points


def get_probability_space(likelihood, posterior):
    fmean = posterior.mean.squeeze()
    fvar = posterior.variance.squeeze()
    if isinstance(likelihood, BernoulliLikelihood):
        # Probability-space mean and variance for Bernoulli-probit models is
        # available in closed form, Proposition 1 in Letham et al. 2022 (AISTATS).
        a_star = fmean / torch.sqrt(1 + fvar)
        pmean = Normal(0, 1).cdf(a_star)
        t_term = torch.tensor(
            owens_t(a_star.numpy(), 1 / np.sqrt(1 + 2 * fvar.numpy())),
            dtype=a_star.dtype,
        )
        pvar = pmean - 2 * t_term - pmean.square()
    else:
        fsamps = posterior.sample(torch.Size([10000]))
        if hasattr(likelihood, "objective"):
            psamps = likelihood.objective(fsamps)
        else:
            psamps = norm.cdf(fsamps)
        pmean, pvar = psamps.mean(0), psamps.var(0)

    return pmean, pvar


def get_extremum(
    model: Model,
    extremum_type: str,
    bounds: torch.Tensor,
    locked_dims: Optional[Mapping[int, List[float]]],
    n_samples: int,
) -> Tuple[float, np.ndarray]:
    """Return the extremum (min or max) of the modeled function
    Args:
        extremum_type (str): type of extremum (currently 'min' or 'max'
        n_samples int: number of coarse grid points to sample for optimization estimate.
    Returns:
        Tuple[float, np.ndarray]: Tuple containing the min and its location (argmin).
    """
    locked_dims = locked_dims or {}

    acqf = PosteriorMean(model=model, maximize=(extremum_type == "max"))
    best_point, best_val = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=n_samples,
        fixed_features=locked_dims,
    )

    # PosteriorMean flips the sign on minimize, we flip it back
    if extremum_type == "min":
        best_val = -best_val
    return best_val, best_point.squeeze(0)
