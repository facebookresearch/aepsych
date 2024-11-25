#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable
from configparser import NoOptionError
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from aepsych.config import Config
from botorch.models.gpytorch import GPyTorchModel
from scipy.stats import norm
from torch.quasirandom import SobolEngine


def make_scaled_sobol(
    lb: torch.Tensor, ub: torch.Tensor, size: int, seed: Optional[int] = None
) -> torch.Tensor:
    lb, ub, ndim = _process_bounds(lb, ub, None)
    grid = SobolEngine(dimension=ndim, scramble=True, seed=seed).draw(size).to(lb)

    # rescale from [0,1] to [lb, ub]
    grid = lb + (ub - lb) * grid

    return grid


def promote_0d(x: Union[torch.Tensor, np.ndarray]):
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
            mesh_vals.append(slice(slice_dims[i] - 1e-10, slice_dims[i] + 1e-10, 1))
        else:
            mesh_vals.append(slice(lower[i].item(), upper[i].item(), gridsize * 1j))

    return torch.Tensor(np.mgrid[mesh_vals].reshape(dim, -1).T)


def _process_bounds(
    lb: Union[torch.Tensor, np.ndarray],
    ub: Union[torch.Tensor, np.ndarray],
    dim: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
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
    model: GPyTorchModel,
    mono_grid: Union[torch.Tensor, np.ndarray],
    target_level: float,
    cred_level: Optional[float] = None,
    mono_dim: int = -1,
    n_samps: int = 500,
    lb: float = -float("inf"),
    ub: float = float("inf"),
    gridsize: int = 30,
    **kwargs,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    # Create a meshgrid using torch.linspace
    xgrid = torch.stack(
        torch.meshgrid(
            [
                torch.linspace(model.lb[i].item(), model.ub[i].item(), gridsize)
                for i in range(model.dim)
            ]
        ),
        dim=-1,
    ).reshape(-1, model.dim)

    if model.transforms is not None:
        xgrid = model.transforms.untransform(xgrid)

    samps = model.sample(xgrid, num_samples=n_samps, **kwargs)
    samps = [s.reshape((gridsize,) * model.dim) for s in samps]

    # Define the normal distribution for the CDF
    normal_dist = torch.distributions.Normal(0, 1)

    # Calculate contours using torch.stack and the torch CDF for each sample
    contours = torch.stack(
        [
            get_lse_contour(
                normal_dist.cdf(s), mono_grid, target_level, mono_dim, lb, ub
            )
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


def get_lse_contour(post_mean, mono_grid, level, mono_dim=-1, lb=-np.inf, ub=np.inf):
    return torch.tensor(
        np.apply_along_axis(
            lambda p: interpolate_monotonic(mono_grid, p, level, lb, ub),
            mono_dim,
            post_mean,
        )
    )


def get_jnd_1d(post_mean, mono_grid, df=1, mono_dim=-1, lb=-np.inf, ub=np.inf):
    interpolate_to = post_mean + df
    return torch.tensor(
        (
            np.array(
                [
                    interpolate_monotonic(mono_grid, post_mean, ito)
                    for ito in interpolate_to
                ]
            )
            - mono_grid.numpy()
        )
    )


def get_jnd_multid(
    post_mean: torch.Tensor,
    mono_grid: torch.Tensor,
    df: int = 1,
    mono_dim: int = -1,
    lb: Union[torch.Tensor, float] = -float("inf"),
    ub: Union[torch.Tensor, float] = float("inf"),
) -> torch.Tensor:
    # Move mono_dim to the last dimension if it isn't already
    if mono_dim != -1:
        post_mean = post_mean.transpose(mono_dim, -1)

    # Apply get_jnd_1d in a vectorized way
    result = get_jnd_1d(post_mean, mono_grid, df=df, mono_dim=-1, lb=lb, ub=ub)

    # Transpose back if necessary
    if mono_dim != -1:
        result = result.transpose(-1, mono_dim)

    return result


def get_bounds(config: Config) -> torch.Tensor:
    r"""Return the bounds for all parameters in config.

    Args:
        config (Config): The config to find the bounds from.

    Returns:
        torch.Tensor: A `[2, d]` tensor with the lower and upper bounds for each
            parameter.
    """
    parnames = config.getlist("common", "parnames", element_type=str)

    # Try to build a full array of bounds based on parameter-specific bounds
    try:
        _lower_bounds = torch.tensor(
            [config.getfloat(par, "lower_bound") for par in parnames]
        )
        _upper_bounds = torch.tensor(
            [config.getfloat(par, "upper_bound") for par in parnames]
        )

        bounds = torch.stack((_lower_bounds, _upper_bounds))

    except NoOptionError:  # Look for general lb/ub array
        _lb = config.gettensor("common", "lb")
        _ub = config.gettensor("common", "ub")
        bounds = torch.stack((_lb, _ub))

    return bounds


def get_optimizer_options(config: Config, name: str) -> Dict[str, Any]:
    """Return the optimizer options for the model to pass to the SciPy L-BFGS-B
    optimizer. Only the somewhat useful ones for AEPsych are searched for: maxcor,
    ftol, gtol, maxfun, maxiter, maxls. See docs for details:
    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb

    Args:
        config (Config): Config to search for options.
        name (str): Model name to look for options for.

    Return:
        Dict[str, Any]: Dictionary of options to pass to SciPy's minimize, assuming the
            method is L-BFGS-B.
    """
    options: Dict[str, Optional[Union[float, int]]] = {}

    options["maxcor"] = config.getint(name, "maxcor", fallback=None)
    options["ftol"] = config.getfloat(name, "ftol", fallback=None)
    options["gtol"] = config.getfloat(name, "gtol", fallback=None)
    options["maxfun"] = config.getint(name, "maxfun", fallback=None)
    options["maxiter"] = config.getint(name, "maxiter", fallback=None)
    options["maxls"] = config.getint(name, "maxls", fallback=None)

    # Filter all the nones out, which could just come back as an empty dict
    options = {key: value for key, value in options.items() if value is not None}
    return options
