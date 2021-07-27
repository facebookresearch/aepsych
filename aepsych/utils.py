#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable

import numpy as np
import torch
from scipy.stats import norm
from torch.quasirandom import SobolEngine


def make_scaled_sobol(lb, ub, size, seed=None):
    # coerce sloppy inputs, this will fail mysteriously and should be less dumb
    if type(lb) != np.ndarray:
        lb = np.r_[lb]
    if type(ub) != np.ndarray:
        ub = np.r_[ub]
    assert lb.shape == ub.shape, "lower and upper bounds should be same shape!"
    ndim = len(lb)
    if seed is not None:
        grid = SobolEngine(dimension=ndim, scramble=True, seed=seed).draw(size)
    else:
        # TODO once https://github.com/pytorch/pytorch/issues/36341 is resolved
        # this randint seed is not needed, but until then this is needed
        # for reproducibility
        seed = torch.randint(high=int(1e6), size=(1,)).item()
        grid = SobolEngine(dimension=ndim, scramble=True, seed=seed).draw(size)

    # rescale from [0,1] to [lb, ub]
    grid = lb + (ub - lb) * grid.numpy()

    return grid


def promote_0d(x):
    if not isinstance(x, Iterable):
        return [x]
    return x


def _dim_grid(modelbridge=None, lower=None, upper=None, dim=None, gridsize=30):

    """Create a grid
    Create a grid based on either modelbridge dimensions, or pass in lower, upper, and dim separately.
    Parameters
    ----------
    modelbridge : ModelBridge
        Input ModelBridge object that defines:
        - lower ('int') - lower bound
        - upper ('int') - upper bound
        - dim ('int) - dimension
    - lower ('int') - lower bound
    - upper ('int') - upper bound
    - dim ('int) - dimension
    - gridsize ('int') - size for grid
    Returns
    ----------
    grid : torch.FloatTensor
        Tensor
    """

    from aepsych.modelbridge.base import ModelBridge

    if modelbridge:
        assert isinstance(modelbridge, ModelBridge), "Not a ModelBridge"
        lower = modelbridge.lb
        upper = modelbridge.ub
        dim = modelbridge.dim
    else:
        assert None not in (lower, upper, dim), (
            "Either pass in lower, upper, and dim, or just "
            "pass modelbridge (but not both)."
        )

    lower = torch.Tensor(promote_0d(lower)).float()
    upper = torch.Tensor(promote_0d(upper)).float()

    return torch.Tensor(
        np.mgrid[
            [slice(lower[i].item(), upper[i].item(), gridsize * 1j) for i in range(dim)]
        ]
        .reshape(dim, -1)
        .T
    )


def interpolate_monotonic(x, y, z, min_x=-np.inf, max_x=np.inf):
    # Ben Letham's 1d interpolation code, assuming monotonicity.
    # basic idea is find the nearest two points to the LSE and
    # linearly interpolate between them (I think this is bisection
    # root-finding)
    idx = np.searchsorted(y, z)
    if idx == len(y):
        return float(max_x)
    elif idx == 0:
        return float(min_x)
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]

    x_star = x0 + (x1 - x0) * (z - y0) / (y1 - y0)
    return x_star


def get_lse_interval(
    modelbridge,
    mono_grid,
    target_level,
    cred_level=None,
    mono_dim=-1,
    n_samps=500,
    lb=-np.inf,
    ub=np.inf,
    gridsize=30,
    **kwargs
):

    xgrid = torch.Tensor(
        np.mgrid[
            [
                slice(modelbridge.lb[i].item(), modelbridge.ub[i].item(), gridsize * 1j)
                for i in range(modelbridge.dim)
            ]
        ]
        .reshape(modelbridge.dim, -1)
        .T
    )

    samps = modelbridge.sample(xgrid, num_samples=n_samps, **kwargs)
    samps = [s.reshape((gridsize,) * modelbridge.dim) for s in samps.detach().numpy()]
    contours = np.stack(
        [
            get_lse_contour(norm.cdf(s), mono_grid, target_level, mono_dim, lb, ub)
            for s in samps
        ]
    )

    if cred_level is None:
        return np.mean(contours, 0.5, axis=0)
    else:

        alpha = 1 - cred_level
        qlower = alpha / 2
        qupper = 1 - alpha / 2

        upper = np.quantile(contours, qupper, axis=0)
        lower = np.quantile(contours, qlower, axis=0)
        median = np.quantile(contours, 0.5, axis=0)

        return median, lower, upper


def get_lse_contour(post_mean, mono_grid, level, mono_dim=-1, lb=-np.inf, ub=np.inf):
    return np.apply_along_axis(
        lambda p: interpolate_monotonic(mono_grid, p, level, lb, ub),
        mono_dim,
        post_mean,
    )


def get_jnd_1d(post_mean, mono_grid, df=1, mono_dim=-1, lb=-np.inf, ub=np.inf):
    interpolate_to = post_mean + df
    return (
        np.array(
            [interpolate_monotonic(mono_grid, post_mean, ito) for ito in interpolate_to]
        )
        - mono_grid
    )


def get_jnd_multid(post_mean, mono_grid, df=1, mono_dim=-1, lb=-np.inf, ub=np.inf):
    return np.apply_along_axis(
        lambda p: get_jnd_1d(p, mono_grid, df=df, mono_dim=mono_dim, lb=lb, ub=ub),
        mono_dim,
        post_mean,
    )
