#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from typing import Mapping

import numpy as np
import torch
from aepsych.config import Config
from aepsych.models.base import AEPsychModelMixin
from aepsych.utils import dim_grid, get_jnd_multid
from botorch.acquisition import PosteriorMean
from botorch.acquisition.objective import (
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import BernoulliLikelihood, Likelihood
from scipy.special import owens_t
from scipy.stats import norm
from torch.distributions import Normal


def compute_p_quantile(
    f_mean: torch.Tensor, f_std: torch.Tensor, alpha: torch.Tensor | float
) -> torch.Tensor:
    """Compute quantile of p in probit model

    For f ~ N(mu_f, sigma_f^2) and p = Phi(f), computes the alpha quantile of p
    using the formula

    x = Phi(mu_f + Phi^-1(alpha) * sigma_f),

    which solves for x such that P(p <= x) = alpha.

    A 95% CI for p can be computed as
    p_l = compute_p_quantile(f_mean, f_std, 0.025)
    p_u = compute_p_quantile(f_mean, f_std, 0.975)

    Args:
        f_mean (torch.Tensor): The mean of the latent function.
        f_std (torch.Tensor): The standard deviation of the latent function.
        alpha (torch.Tensor | float): The quantile to compute.

    Returns:
        torch.Tensor: The quantile of p.
    """
    norm = torch.distributions.Normal(0, 1)
    alpha = torch.tensor(alpha, dtype=f_mean.dtype)
    return norm.cdf(f_mean + norm.icdf(alpha) * f_std)


def get_probability_space(
    likelihood: Likelihood, posterior: GPyTorchPosterior
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the mean and variance of the probability space for a given posterior

    Args:
        likelihood (Likelihood): The likelihood function.
        posterior (GPyTorchPosterior): The posterior to transform.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The mean and variance of the probability space.
    """
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
    locked_dims: Mapping[int, float] | None,
    n_samples: int,
    posterior_transform: PosteriorTransform | None = None,
    max_time: float | None = None,
    weights: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor]:
    """Return the extremum (min or max) of the modeled function
    Args:
        extremum_type (str): Type of extremum (currently 'min' or 'max'.
        bounds (torch.Tensor): Lower and upper bounds of the search space.
        locked_dims (Mapping[int, float], optional): Dimensions to fix, so that the
            extremum is along a slice of the full surface.
        n_samples (int): number of coarse grid points to sample for optimization estimate.
        posterior_transform (PosteriorTransform, optional): Posterior transform to apply to the model.
        max_time (float, optional): Maximum amount of time in seconds to spend optimizing.
        weights (torch.Tensor, optional): Weights to apply to the target value. Defaults to None.
    Returns:
        tuple[float, torch.Tensor]: Tuple containing the min and its location (argmin).
    """
    locked_dims = locked_dims or {}

    if hasattr(model, "transforms"):
        # Transform locked dims
        tmp = {}
        for key, value in locked_dims.items():
            tensor = torch.zeros(model.dim)
            tensor[key] = value
            tensor = model.transforms.transform(tensor)
            tmp[key] = tensor[key].item()
        locked_dims = tmp

        # Transform bounds
        bounds = model.transforms.transform(bounds)

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

    if hasattr(model, "transforms"):
        best_point = model.transforms.untransform(best_point)

    # PosteriorMean flips the sign on minimize, we flip it back
    if extremum_type == "min":
        best_val = -best_val
    return best_val, best_point.squeeze(0)


def get_min(
    model: AEPsychModelMixin,
    bounds: torch.Tensor,
    locked_dims: Mapping[int, float] | None = None,
    probability_space: bool = False,
    n_samples: int = 1000,
    max_time: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the minimum of the modeled function, subject to constraints
    Args:
        model (AEPsychModelMixin): AEPsychModel to get the minimum of.
        bounds (torch.Tensor): Bounds of the space to find the minimum.
        locked_dims (Mapping[int, float], optional): Dimensions to fix, so that the
            inverse is along a slice of the full surface.
        probability_space (bool): Is y (and therefore the returned nearest_y) in
            probability space instead of latent function space? Defaults to False.
        n_samples (int): number of coarse grid points to sample for optimization estimate.
        max_time (float, optional): Maximum time to spend optimizing. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple containing the min and its location (argmin).
    """
    _, _arg = get_extremum(
        model, "min", bounds, locked_dims, n_samples, max_time=max_time
    )
    arg = _arg.reshape(1, bounds.shape[1])
    if probability_space:
        val, _ = model.predict_probability(arg)
    else:
        val, _ = model.predict(arg)

    return val.detach(), arg


def get_max(
    model: AEPsychModelMixin,
    bounds: torch.Tensor,
    locked_dims: Mapping[int, float] | None = None,
    probability_space: bool = False,
    n_samples: int = 1000,
    max_time: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the maximum of the modeled function, subject to constraints

    Args:
        model (AEPsychModelMixin): AEPsychModel to get the maximum of.
        bounds (torch.Tensor): Bounds of the space to find the maximum.
        locked_dims (Mapping[int, float], optional): Dimensions to fix, so that the
            inverse is along a slice of the full surface. Defaults to None.
        probability_space (bool): Is y (and therefore the returned nearest_y) in
            probability space instead of latent function space? Defaults to False.
        n_samples (int): number of coarse grid points to sample for optimization estimate.
        max_time (float, optional): Maximum time to spend optimizing. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple containing the max and its location (argmax).
    """
    _, _arg = get_extremum(
        model, "max", bounds, locked_dims, n_samples, max_time=max_time
    )
    arg = _arg.reshape(1, bounds.shape[1])
    if probability_space:
        val, _ = model.predict_probability(arg)
    else:
        val, _ = model.predict(arg)

    return val.detach(), arg


def inv_query(
    model: AEPsychModelMixin,
    y: float | torch.Tensor,
    bounds: torch.Tensor,
    locked_dims: Mapping[int, float] | None = None,
    probability_space: bool = False,
    n_samples: int = 1000,
    max_time: float | None = None,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Query the model inverse.
    Return nearest x such that f(x) = queried y, and also return the
        value of f at that point.
    Args:
        model (AEPsychModelMixin): AEPsychModel to get the find the inverse from y.
        y (float | torch.Tensor): Points at which to find the inverse.
        bounds (torch.Tensor): Lower and upper bounds of the search space.
        locked_dims (Mapping[int, float], optional): Dimensions to fix, so that the
            inverse is along a slice of the full surface. Defaults to None.
        probability_space (bool): Is y (and therefore the
            returned nearest_y) in probability space instead of latent
            function space? Defaults to False.
        n_samples (int): number of coarse grid points to sample for optimization estimate. Defaults to 1000.
        max_time (float, optional): Maximum amount of time in seconds to spend optimizing. Defaults to None.
        weights (torch.Tensor, optional): Weights to apply to the target value. Defaults to None.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple containing the value of f
            nearest to queried y and the x position of this value.
    """
    locked_dims = locked_dims or {}
    if model._num_outputs > 1:
        if weights is None:
            weights = torch.Tensor([1] * model._num_outputs)
    if probability_space:
        warnings.warn(
            "Inverse querying with probability_space=True assumes that the model uses Probit-Bernoulli likelihood!",
            stacklevel=2,
        )
        posterior_transform = TargetProbabilityDistancePosteriorTransform(y, weights)
    else:
        posterior_transform = TargetDistancePosteriorTransform(y, weights)
    _, _arg = get_extremum(
        model,
        "min",
        bounds,
        locked_dims,
        n_samples,
        posterior_transform,
        max_time,
        weights,
    )

    arg = _arg.reshape(1, bounds.shape[1])
    if probability_space:
        val, _ = model.predict_probability(arg)
    else:
        val, _ = model.predict(arg)

    return val.detach(), arg


def get_jnd(
    model: AEPsychModelMixin,
    lb: torch.Tensor,
    ub: torch.Tensor,
    dim: int,
    grid: np.ndarray | torch.Tensor | None = None,
    cred_level: float | None = None,
    intensity_dim: int = -1,
    confsamps: int = 500,
    method: str = "step",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the JND.

    Note that JND can have multiple plausible definitions
    outside of the linear case, so we provide options for how to compute it.
    For method="step", we report how far one needs to go over in stimulus
    space to move 1 unit up in latent space (this is a lot of people's
    conventional understanding of the JND).
    For method="taylor", we report the local derivative, which also maps to a
    1st-order Taylor expansion of the latent function. This is a formal
    generalization of JND as defined in Weber's law.
    Both definitions are equivalent for linear psychometric functions.

    Args:
        model (AEPsychModelMixin): Model to use for prediction.
        lb (torch.Tensor): Lower bounds of the input space.
        ub (torch.Tensor): Upper bounds of the input space.
        dim (int): Dimensionality of the input space.
        grid (np.ndarray | torch.Tensor, optional): Mesh grid over which to find the JND.
            Defaults to a square grid of size as determined by aepsych.utils.dim_grid
        cred_level (float, optional): Credible level for computing an interval.
            Defaults to None, computing no interval.
        intensity_dim (int, optional): Dimension over which to compute the JND.
            Defaults to -1.
        confsamps (int, optional): Number of posterior samples to use for
            computing the credible interval. Defaults to 500.
        method (str, optional): "taylor" or "step" method (see docstring).
            Defaults to "step".

    Raises:
        RuntimeError: for passing an unknown method.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]: either the
            mean JND, or a median, lower, upper tuple of the JND posterior.
    """
    if grid is None:
        grid = dim_grid(lower=lb, upper=ub, gridsize=30, slice_dims=None)
    elif isinstance(grid, np.ndarray):
        grid = torch.tensor(grid)

    # this is super awkward, back into intensity dim grid assuming a square grid
    gridsize = int(grid.shape[0] ** (1 / grid.shape[1]))
    coords = torch.linspace(
        lb[intensity_dim].item(), ub[intensity_dim].item(), gridsize
    )

    if cred_level is None:
        fmean, _ = model.predict(grid)
        fmean = fmean.reshape(*[gridsize for i in range(dim)])

        if method == "taylor":
            return torch.tensor(1 / np.gradient(fmean, coords, axis=intensity_dim))
        elif method == "step":
            return torch.clip(
                get_jnd_multid(
                    fmean,
                    coords,
                    mono_dim=intensity_dim,
                ),
                0,
                np.inf,
            )

    alpha = 1 - cred_level  # type: ignore
    qlower = alpha / 2
    qupper = 1 - alpha / 2

    fsamps = model.sample(grid, confsamps)
    if method == "taylor":
        jnds = torch.tensor(
            1
            / np.gradient(
                fsamps.reshape(confsamps, *[gridsize for i in range(dim)]),
                coords,
                axis=intensity_dim,
            )
        )
    elif method == "step":
        samps = [s.reshape((gridsize,) * dim) for s in fsamps]
        jnds = torch.stack(
            [get_jnd_multid(s, coords, mono_dim=intensity_dim) for s in samps]
        )
    else:
        raise RuntimeError(f"Unknown method {method}!")
    upper = torch.clip(torch.quantile(jnds, qupper, axis=0), 0, np.inf)  # type: ignore
    lower = torch.clip(torch.quantile(jnds, qlower, axis=0), 0, np.inf)  # type: ignore
    median = torch.clip(torch.quantile(jnds, 0.5, axis=0), 0, np.inf)  # type: ignore
    return median, lower, upper


def p_below_threshold(
    model: AEPsychModelMixin, x: torch.Tensor, f_thresh: torch.Tensor
) -> torch.Tensor:
    """Compute the probability that the latent function is below a threshold.

    Args:
        x (torch.Tensor): Points at which to evaluate the probability.
        f_thresh (torch.Tensor): Threshold value.

    Returns:
        torch.Tensor: Probability that the latent function is below the threshold.
    """
    f, var = model.predict(x)
    f_thresh = f_thresh.reshape(-1, 1)
    f = f.reshape(1, -1)
    var = var.reshape(1, -1)

    z = (f_thresh - f) / var.sqrt()
    return torch.distributions.Normal(0, 1).cdf(z)  # Use PyTorch's CDF equivalent


def bernoulli_probit_prob_transform(mean: torch.Tensor, var: torch.Tensor):
    """Computes the probability-space mean and variance from a Bernoulli-probit model. Closed form expression taken from Proposition 1 in Letham et al. 2022 (AISTATS).

    Args:
        mean (torch.Tensor): The latent mean of a Bernoulli-probit model evaluated at a set of query points.
        mean (torch.Tensor): The latent variance of a Bernoulli-probit model evaluated at a set of query points.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at query points in probability space.
    """
    fmean = mean.squeeze()
    fvar = var.squeeze()
    a_star = fmean / torch.sqrt(1 + fvar)
    pmean = Normal(0, 1).cdf(a_star)
    t_term = torch.tensor(
        owens_t(a_star.cpu().numpy(), 1 / np.sqrt(1 + 2 * fvar.cpu().numpy())),
        dtype=a_star.dtype,
    ).to(a_star)
    pvar = pmean - 2 * t_term - pmean.square()
    return pmean, pvar


class TargetDistancePosteriorTransform(PosteriorTransform):
    def __init__(
        self,
        target_value: float | torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        """Initialize the TargetDistancePosteriorTransform

        Args:
            target_value (float | torch.Tensor): The target value to transform the posterior to.
            weights (torch.Tensor, optional): Weights to apply to the target value. Defaults to None.
        """
        super().__init__()
        self.target_value = target_value
        self.weights = weights

    def evaluate(self, Y: torch.Tensor) -> torch.Tensor:
        """Evaluate the squared distance from the target value.

        Args:
            Y (torch.Tensor): The tensor to evaluate.

        Returns:
            torch.Tensor: The squared distance from the target value.
        """
        return (Y - self.target_value) ** 2

    def _forward(self, mean: torch.Tensor, var: torch.Tensor) -> GPyTorchPosterior:
        """Transform the posterior mean and variance based on the target value.

        Args:
            mean (torch.Tensor): The posterior mean.
            var (torch.Tensor): The posterior variance.

        Returns:
            GPyTorchPosterior: The transformed posterior.
        """
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
        """Transform the given posterior distribution to reflect the target distance.

        Args:
            posterior (GPyTorchPosterior): The posterior to transform.

        Returns:
            GPyTorchPosterior: The transformed posterior.
        """
        mean = posterior.mean
        var = posterior.variance
        return self._forward(mean, var)


# Requires botorch approximate model to accept posterior transforms
class TargetProbabilityDistancePosteriorTransform(TargetDistancePosteriorTransform):
    def forward(self, posterior: GPyTorchPosterior) -> GPyTorchPosterior:
        """Transform the given posterior distribution to reflect the target probability distance.

        Args:
            posterior (GPyTorchPosterior): The posterior to transform.

        Returns:
            GPyTorchPosterior: The transformed posterior distribution reflecting the target probability distance.
        """
        pmean, pvar = get_probability_space(BernoulliLikelihood(), posterior)
        pmean = pmean.unsqueeze(-1).unsqueeze(-1)
        pvar = pvar.unsqueeze(-1).unsqueeze(-1)
        return self._forward(pmean, pvar)


def constraint_factory(
    config: Config,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Given a config, return a set of constraint locations and corresponding
    values. The config should include the constraint bounds/values and how many
    points to place in the bounds with that value.

    Args:
        config (Config): A config object containing the constraint information.
            Must include constraint_lower, constraint_upper,
            constraint_values, and optionally constraint_points and
            constraint_strengths.

    Returns:
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]: A tuple
            containing the constraint locations, values, and strengths. Tuple can be
            all None if points is set to 0.
    """
    lowers = config.gettensor("constraint_factory", "constraint_lower")
    uppers = config.gettensor("constraint_factory", "constraint_upper")
    values = config.gettensor("constraint_factory", "constraint_values")
    strengths = config.gettensor(
        "constraint_factory", "constraint_strengths", fallback=None
    )
    points = config.getint("constraint_factory", "points_per_dim", fallback=10)

    if points == 0:
        return None, None, None

    if lowers.ndim == 1:
        lowers = lowers.unsqueeze(0)

    if uppers.ndim == 1:
        uppers = uppers.unsqueeze(0)

    if strengths is None:
        strengths = (0.2 * values + 0.1) ** 2

    loc_list = []
    val_list = []
    str_list = []

    # For each constraint, create a uniform grid of points between the bounds
    for lower, upper, value, strength in zip(lowers, uppers, values, strengths):
        grid = dim_grid(lower=lower, upper=upper, gridsize=points)
        grid = grid.unique(dim=0)

        str_list.append(strength.repeat(grid.shape[0], 1))
        loc_list.append(grid)
        val_list.append(value.repeat(grid.shape[0], 1))

    locs = torch.cat(loc_list, dim=0)
    vals = torch.cat(val_list, dim=0)
    strs = torch.cat(str_list, dim=0)

    # Deduplicate points
    tmp = torch.cat([locs, vals, strs], dim=1)
    tmp = tmp.unique(dim=0)
    locs = tmp[:, : locs.shape[1]]
    vals = tmp[:, locs.shape[1] : locs.shape[1] + vals.shape[1]]
    strs = tmp[:, locs.shape[1] + vals.shape[1] :]

    return locs, vals, strs
