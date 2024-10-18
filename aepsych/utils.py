#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable
from configparser import NoOptionError
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import norm
from torch.quasirandom import SobolEngine

from aepsych.config import Config

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

    lower, upper, dim = _process_bounds(lower, upper, None)

    mesh_vals = []

    for i in range(dim):
        if i in slice_dims.keys():
            mesh_vals.append(torch.tensor([slice_dims[i] - 1e-10, slice_dims[i] + 1e-10]))
        else:
            mesh_vals.append(torch.linspace(lower[i].item(), upper[i].item(), gridsize))

    return torch.stack(torch.meshgrid(*mesh_vals, indexing='ij'), dim=-1).reshape(-1, dim)


def _process_bounds(lb, ub, dim) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Helper function for ensuring bounds are correct shape and type."""
    lb = promote_0d(lb)
    ub = promote_0d(ub)

    if not isinstance(lb, torch.Tensor):
        lb = torch.tensor(lb)
    if not isinstance(ub, torch.Tensor):
        ub = torch.tensor(ub)

    lb = lb.to(torch.float64)
    ub = ub.to(torch.float64)

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


def interpolate_monotonic(x: torch.Tensor, y: torch.Tensor, z: Union[torch.Tensor, float], min_x: Union[torch.Tensor, float] =-float('inf'), max_x: Union[torch.Tensor, float] =float('inf')) -> torch.Tensor:
    # Ben Letham's 1d interpolation code, assuming monotonicity.
    # basic idea is find the nearest two points to the LSE and
    # linearly interpolate between them (I think this is bisection
    # root-finding)
    idx = torch.searchsorted(y, z, right=False)
    
    # Handle edge cases where idx is 0 or at the end
    idx = torch.clamp(idx, 1, len(y) - 1)
    
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]

    x_star = x0 + (x1 - x0) * (z - y0) / (y1 - y0)
    # Apply min and max boundaries
    x_star = torch.where(z < y[0], min_x, x_star)
    x_star = torch.where(z > y[-1], max_x, x_star)
    
    return x_star


def get_lse_interval(
    model,
    mono_grid: Union[torch.Tensor, np.ndarray],
    target_level: float,
    cred_level: Optional[float]=None,
    mono_dim: int =-1,
    n_samps: int =500,
    lb: float =-float('inf'),
    ub: float =float('inf'),
    gridsize: int =30,
    **kwargs,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    # Create a meshgrid using torch.linspace
    xgrid = torch.stack(
        torch.meshgrid(
            [torch.linspace(model.lb[i].item(), model.ub[i].item(), gridsize) for i in range(model.dim)]
        ),
        dim=-1
    ).reshape(-1, model.dim)

    samps = model.sample(xgrid, num_samples=n_samps, **kwargs)
    samps = [s.reshape((gridsize,) * model.dim) for s in samps]

    # Define the normal distribution for the CDF
    normal_dist = torch.distributions.Normal(0, 1)

    # Calculate contours using torch.stack and the torch CDF for each sample
    contours = torch.stack(
        [
            get_lse_contour(normal_dist.cdf(s), mono_grid, target_level, mono_dim, lb, ub)
            for s in samps
        ]
    )

    if cred_level is None:
        return torch.median(contours, dim=0).values
    else:
        alpha = 1 - cred_level
        qlower = alpha / 2
        qupper = 1 - alpha / 2

        lower = torch.quantile(contours, qlower, dim=0)
        upper = torch.quantile(contours, qupper, dim=0)
        median = torch.quantile(contours, 0.5, dim=0)

        return median, lower, upper


def get_lse_contour(post_mean: torch.Tensor, mono_grid: Union[torch.Tensor, np.ndarray], level: float, mono_dim: int =-1, lb: Union[torch.Tensor, float] =-float('inf'), ub: Union[torch.Tensor, float] =float('inf')) -> torch.Tensor:
    post_mean = torch.tensor(post_mean, dtype=torch.float32)
    mono_grid = torch.tensor(mono_grid, dtype=torch.float32)
    
    # Move mono_dim to the last dimension if it isn't already
    if mono_dim != -1:
        post_mean = post_mean.transpose(mono_dim, -1)
    
    # Apply interpolation across all rows at once
    result = interpolate_monotonic(mono_grid, post_mean, level, lb, ub)
    
    # Transpose back if necessary
    if mono_dim != -1:
        result = result.transpose(-1, mono_dim)
    
    return result


def get_jnd_1d(post_mean: torch.Tensor, mono_grid: torch.Tensor, df: int =1, mono_dim: int =-1, lb: Union[torch.Tensor, float] =-float('inf'), ub: Union[torch.Tensor, float] =float('inf')) -> torch.Tensor:
    
    # Calculate interpolate_to in a vectorized way
    interpolate_to = post_mean + df
    
    # Apply interpolation to the entire tensor
    interpolated_values = interpolate_monotonic(mono_grid, post_mean, interpolate_to, lb, ub)
    
    return interpolated_values - mono_grid

def get_jnd_multid(post_mean: torch.Tensor, mono_grid: torch.Tensor, df: int =1, mono_dim: int =-1, lb: Union[torch.Tensor, float] =-float('inf'), ub: Union[torch.Tensor, float] =float('inf')) -> torch.Tensor:
    
    # Move mono_dim to the last dimension if it isn't already
    if mono_dim != -1:
        post_mean = post_mean.transpose(mono_dim, -1)
    
    # Apply get_jnd_1d in a vectorized way
    result = get_jnd_1d(post_mean, mono_grid, df=df, mono_dim=-1, lb=lb, ub=ub)
    
    # Transpose back if necessary
    if mono_dim != -1:
        result = result.transpose(-1, mono_dim)
    
    return result


def _get_ax_parameters(config):
    range_parnames = config.getlist("common", "parnames", element_type=str, fallback=[])
    lb = config.getlist("common", "lb", element_type=float, fallback=[])
    ub = config.getlist("common", "ub", element_type=float, fallback=[])

    assert (
        len(range_parnames) == len(lb) == len(ub)
    ), f"Length of parnames ({range_parnames}), lb ({lb}), and ub ({ub}) don't match!"

    range_params = [
        {
            "name": parname,
            "type": "range",
            "value_type": config.get(parname, "value_type", fallback="float"),
            "log_scale": config.getboolean(parname, "log_scale", fallback=False),
            "bounds": [l, u],
        }
        for parname, l, u in zip(range_parnames, lb, ub)
    ]

    choice_parnames = config.getlist(
        "common", "choice_parnames", element_type=str, fallback=[]
    )
    choices = [
        config.getlist(parname, "choices", element_type=str, fallback=["True", "False"])
        for parname in choice_parnames
    ]
    choice_params = [
        {
            "name": parname,
            "type": "choice",
            "value_type": config.get(parname, "value_type", fallback="str"),
            "is_ordered": config.getboolean(parname, "is_ordered", fallback=False),
            "values": choice,
        }
        for parname, choice in zip(choice_parnames, choices)
    ]

    fixed_parnames = config.getlist(
        "common", "fixed_parnames", element_type=str, fallback=[]
    )
    values = []
    for parname in fixed_parnames:
        try:
            try:
                value = config.getfloat(parname, "value")
            except ValueError:
                value = config.get(parname, "value")
            values.append(value)
        except NoOptionError:
            raise RuntimeError(f"Missing value for fixed parameter {parname}!")
    fixed_params = [
        {
            "name": parname,
            "type": "fixed",
            "value": value,
        }
        for parname, value in zip(fixed_parnames, values)
    ]

    return range_params, choice_params, fixed_params


def get_parameters(config) -> List[Dict]:
    range_params, choice_params, fixed_params = _get_ax_parameters(config)
    return range_params + choice_params + fixed_params


def get_bounds(config) -> torch.Tensor:
    range_params, choice_params, _ = _get_ax_parameters(config)
    # Need to sum dimensions added by both range and choice parameters
    bounds = [parm["bounds"] for parm in range_params]
    for par in choice_params:
        n_vals = len(par["values"])
        if par["is_ordered"]:
            bounds.append(
                [0, 1]
            )  # Ordered choice params are encoded like continuous parameters
        elif n_vals > 2:
            for _ in range(n_vals):
                bounds.append(
                    [0, 1]
                )  # Choice parameter is one-hot encoded such that they add 1 dim for every choice
        else:
            for _ in range(n_vals - 1):
                bounds.append(
                    [0, 1]
                )  # Choice parameters with n_choices <= 2 add n_choices - 1 dims

    return torch.tensor(bounds)


def get_dim(config) -> int:
    range_params, choice_params, _ = _get_ax_parameters(config)
    # Need to sum dimensions added by both range and choice parameters
    dim = len(range_params)  # 1 dim per range parameter
    for par in choice_params:
        if par["is_ordered"]:
            dim += 1  # Ordered choice params are encoded like continuous parameters
        elif len(par["values"]) > 2:
            dim += len(
                par["values"]
            )  # Choice parameter is one-hot encoded such that they add 1 dim for every choice
        else:
            dim += (
                len(par["values"]) - 1
            )  # Choice parameters with n_choices < 3 add n_choices - 1 dims

    return dim
