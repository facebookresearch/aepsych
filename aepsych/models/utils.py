#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings

from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.objective import PosteriorTransform
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.model import Model
from botorch.models.utils.inducing_point_allocators import GreedyVarianceReduction
from botorch.optim import optimize_acqf
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from scipy.cluster.vq import kmeans2
from scipy.special import owens_t
from scipy.stats import norm
from torch import Tensor
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
            owens_t(
                a_star.detach().numpy(), 1 / np.sqrt(1 + 2 * fvar.detach().numpy())
            ),
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
    posterior_transform: Optional[PosteriorTransform] = None,
    max_time: Optional[float] = None,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[float, torch.Tensor]:
    """Return the extremum (min or max) of the modeled function
    Args:
        extremum_type (str): Type of extremum (currently 'min' or 'max'.
        bounds (tensor): Lower and upper bounds of the search space.
        locked_dims (Mapping[int, List[float]]): Dimensions to fix, so that the
            inverse is along a slice of the full surface.
        n_samples (int): number of coarse grid points to sample for optimization estimate.
        max_time (float): Maximum amount of time in seconds to spend optimizing.
    Returns:
        Tuple[float, np.ndarray]: Tuple containing the min and its location (argmin).
    """
    locked_dims = locked_dims or {}

    if model.num_outputs > 1 and posterior_transform is None:
        if weights is None:
            weights = torch.Tensor([1] * model.num_outputs)
        posterior_transform = ScalarizedPosteriorTransform(weights=weights)

    acqf = PosteriorMean(
        model=model,
        posterior_transform=posterior_transform,
        maximize=(extremum_type == "max"),
    )
    best_point, best_val = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=n_samples,
        fixed_features=locked_dims,
        timeout_sec=max_time,
    )

    # PosteriorMean flips the sign on minimize, we flip it back
    if extremum_type == "min":
        best_val = -best_val
    return best_val, best_point.squeeze(0)


def inv_query(
    model: Model,
    y: Union[float, torch.Tensor],
    bounds: torch.Tensor,
    locked_dims: Optional[Mapping[int, List[float]]] = None,
    probability_space: bool = False,
    n_samples: int = 1000,
    max_time: Optional[float] = None,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[float, torch.Tensor]:
    """Query the model inverse.
    Return nearest x such that f(x) = queried y, and also return the
        value of f at that point.
    Args:
        y (float): Points at which to find the inverse.
        bounds (tensor): Lower and upper bounds of the search space.
        locked_dims (Mapping[int, List[float]]): Dimensions to fix, so that the
            inverse is along a slice of the full surface.
        probability_space (bool): Is y (and therefore the
            returned nearest_y) in probability space instead of latent
            function space? Defaults to False.
        n_samples (int): number of coarse grid points to sample for optimization estimate.
        max_time float: Maximum amount of time in seconds to spend optimizing.
    Returns:
        Tuple[float, np.ndarray]: Tuple containing the value of f
            nearest to queried y and the x position of this value.
    """
    locked_dims = locked_dims or {}
    if model.num_outputs > 1:
        if weights is None:
            weights = torch.Tensor([1] * model.num_outputs)
    if probability_space:
        warnings.warn(
            "Inverse querying with probability_space=True assumes that the model uses Probit-Bernoulli likelihood!"
        )
        posterior_transform = TargetProbabilityDistancePosteriorTransform(y, weights)
    else:
        posterior_transform = TargetDistancePosteriorTransform(y, weights)
    val, arg = get_extremum(
        model,
        "min",
        bounds,
        locked_dims,
        n_samples,
        posterior_transform,
        max_time,
        weights,
    )
    return val, arg


class TargetDistancePosteriorTransform(PosteriorTransform):
    def __init__(
        self, target_value: Union[float, Tensor], weights: Optional[Tensor] = None
    ):
        super().__init__()
        self.target_value = target_value
        self.weights = weights

    def evaluate(self, Y: Tensor) -> Tensor:
        return (Y - self.target_value) ** 2

    def _forward(self, mean, var):
        q, _ = mean.shape[-2:]
        batch_shape = mean.shape[:-2]

        new_mean = (mean - self.target_value) ** 2

        if self.weights is not None:
            new_mean = new_mean @ self.weights
            var = (var @ (self.weights**2))[:, None]

        new_mean = new_mean.view(*batch_shape, q)
        mvn = MultivariateNormal(new_mean, var)
        return GPyTorchPosterior(mvn)

    def forward(self, posterior: GPyTorchPosterior) -> GPyTorchPosterior:
        mean = posterior.mean
        var = posterior.variance
        return self._forward(mean, var)


# Requires botorch approximate model to accept posterior transforms
class TargetProbabilityDistancePosteriorTransform(TargetDistancePosteriorTransform):
    def forward(self, posterior: GPyTorchPosterior) -> GPyTorchPosterior:
        pmean, pvar = get_probability_space(BernoulliLikelihood(), posterior)
        pmean = pmean.unsqueeze(-1).unsqueeze(-1)
        pvar = pvar.unsqueeze(-1).unsqueeze(-1)
        return self._forward(pmean, pvar)
