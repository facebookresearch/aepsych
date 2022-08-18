#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable
from typing import Mapping, Optional

import numpy as np
import torch
from scipy.stats import norm
from torch.quasirandom import SobolEngine


def make_scaled_sobol(lb, ub, size, seed=None):
    lb, ub, ndim = _process_bounds(lb, ub, None)
    grid = SobolEngine(dimension=ndim, scramble=True, seed=seed).draw(size)

    # rescale from [0,1] to [lb, ub]
    grid = lb + (ub - lb) * grid

    return grid


def promote_0d(x):
    if not isinstance(x, Iterable):
        return [x]
    return x


def dim_grid(
    lower: torch.Tensor,
    upper: torch.Tensor,
    dim: int,
    gridsize: int = 30,
    slice_dims: Optional[Mapping[int, float]] = None,
) -> torch.Tensor:

    """Create a grid
    Create a grid based on lower, upper, and dim.
    Parameters
    ----------
    - lower ('int') - lower bound
    - upper ('int') - upper bound
    - dim ('int) - dimension
    - gridsize ('int') - size for grid
    - slice_dims (Optional, dict) - values to use for slicing axes, as an {index:value} dict
    Returns
    ----------
    grid : torch.FloatTensor
        Tensor
    """
    slice_dims = slice_dims or {}

    lower, upper, _ = _process_bounds(lower, upper, None)

    mesh_vals = []

    for i in range(dim):
        if i in slice_dims.keys():
            mesh_vals.append(slice(slice_dims[i] - 1e-10, slice_dims[i] + 1e-10, 1))
        else:
            mesh_vals.append(slice(lower[i].item(), upper[i].item(), gridsize * 1j))

    return torch.Tensor(np.mgrid[mesh_vals].reshape(dim, -1).T)


def _process_bounds(lb, ub, dim):
    """Helper function for ensuring bounds are correct shape and type."""
    lb = promote_0d(lb)
    ub = promote_0d(ub)

    if not isinstance(lb, torch.Tensor):
        lb = torch.tensor(lb)
    if not isinstance(ub, torch.Tensor):
        ub = torch.tensor(ub)

    lb = lb.float()
    ub = ub.float()
    assert lb.shape[0] == ub.shape[0], "bounds should be of equal shape!"

    if dim is not None:
        if lb.shape[0] == 1:
            lb = lb.repeat(dim)
            ub = ub.repeat(dim)
        else:
            assert lb.shape[0] == dim, "dim does not match shape of bounds!"
    else:
        dim = lb.shape[0]

    for i, (l, u) in enumerate(zip(lb, ub)):
        assert (
            l <= u
        ), f"Lower bound {l} is not less than or equal to upper bound {u} on dimension {i}!"

    return lb, ub, dim


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
    model,
    mono_grid,
    target_level,
    cred_level=None,
    mono_dim=-1,
    n_samps=500,
    lb=-np.inf,
    ub=np.inf,
    gridsize=30,
    **kwargs,
):

    xgrid = torch.Tensor(
        np.mgrid[
            [
                slice(model.lb[i].item(), model.ub[i].item(), gridsize * 1j)
                for i in range(model.dim)
            ]
        ]
        .reshape(model.dim, -1)
        .T
    )

    samps = model.sample(xgrid, num_samples=n_samps, **kwargs)
    samps = [s.reshape((gridsize,) * model.dim) for s in samps.detach().numpy()]
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
