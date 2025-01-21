#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import CubicSpline, interp1d

# manually scraped data from doi:10.1007/s10162-013-0396-x fig 2
raw = """\
freq,thresh,phenotype
0.25,6.816404934,Older-normal
0.5,5.488517768,Older-normal
1,3.512856308,Older-normal
2,5.909671334,Older-normal
3,6.700337017,Older-normal
4,10.08761498,Older-normal
6,13.46962853,Older-normal
8,12.97026073,Older-normal
0.25,5.520856346,Sensory
0.5,4.19296918,Sensory
1,5.618122764,Sensory
2,19.83681866,Sensory
3,42.00403606,Sensory
4,53.32679981,Sensory
6,62.0527006,Sensory
8,66.08775286,Sensory
0.25,21.2291323,Metabolic
0.5,22.00676227,Metabolic
1,24.24163372,Metabolic
2,33.92590956,Metabolic
3,41.35626176,Metabolic
4,47.17294402,Metabolic
6,54.1174655,Metabolic
8,58.31446133,Metabolic
0.25,20.25772154,Metabolic+Sensory
0.5,20.71121368,Metabolic+Sensory
1,21.97442369,Metabolic+Sensory
2,37.48866818,Metabolic+Sensory
3,53.17814263,Metabolic+Sensory
4,64.01507567,Metabolic+Sensory
6,75.00818649,Metabolic+Sensory
8,76.61433583,Metabolic+Sensory"""

dubno_data = pd.read_csv(io.StringIO(raw))


def make_songetal_threshfun(
    x: torch.Tensor, y: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Generate a synthetic threshold function by interpolation of real data.

    Real data is from Dubno et al. 2013, and procedure follows Song et al. 2017, 2018.
    See make_songetal_testfun for more detail.

    Args:
        x (torch.Tensor): Frequency
        y (torch.Tensor): Threshold

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: Function that interpolates the given
            frequencies and thresholds and returns threshold as a function
            of frequency.
    """
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    # These are not directly implemented in pytorch, so we use scipy for now
    f_interp = CubicSpline(x_np, y_np, extrapolate=False)
    f_extrap = interp1d(x_np, y_np, fill_value="extrapolate")

    def f_combo(x):
        x_np = x.cpu().numpy()
        interpolated = f_interp(x_np)
        interpolated[np.isnan(interpolated)] = f_extrap(x_np[np.isnan(interpolated)])
        return torch.from_numpy(interpolated)

    return f_combo


def make_songetal_testfun(
    phenotype: str = "Metabolic", beta: float = 1
) -> Callable[[torch.Tensor, bool], torch.Tensor]:
    """Make an audiometric test function following Song et al. 2017.

    To do so, we first compute a threshold by interpolation/extrapolation
    from real data, then assume a linear psychometric function in intensity
    with slope beta.

    Args:
        phenotype (str, optional): Audiometric phenotype from Dubno et al. 2013.
            Specifically, one of "Metabolic", "Sensory", "Metabolic+Sensory",
            or "Older-normal". Defaults to "Metabolic".
        beta (float, optional): Psychometric function slope. Defaults to 1.

    Returns:
        Callable[[torch.Tensor, bool], torch.Tensor]: A test function taking a [b x 2] tensor of points and returning the psychometric function value at those points.

    Raises:
        AssertionError: if an invalid phenotype is passed.

    References:
        Song, X. D., Garnett, R., & Barbour, D. L. (2017).
            Psychometric function estimation by probabilistic classification.
            The Journal of the Acoustical Society of America, 141(4), 2513–2525.
            https://doi.org/10.1121/1.4979594
    """
    valid_phenotypes = ["Metabolic", "Sensory", "Metabolic+Sensory", "Older-normal"]
    assert phenotype in valid_phenotypes, f"Phenotype must be one of {valid_phenotypes}"
    x = torch.tensor(
        dubno_data[dubno_data.phenotype == phenotype].freq.values, dtype=torch.float64
    )
    y = torch.tensor(
        dubno_data[dubno_data.phenotype == phenotype].thresh.values, dtype=torch.float64
    )

    # first, make the threshold fun
    threshfun = make_songetal_threshfun(x, y)

    # now make it into a test function
    def song_testfun(x, cdf=False):
        logfreq = x[..., 0]
        intensity = x[..., 1]
        thresh = threshfun(2**logfreq)
        return (
            torch.distributions.Normal(0, 1).cdf((intensity - thresh) / beta)
            if cdf
            else (intensity - thresh) / beta
        )

    return song_testfun


def novel_discrimination_testfun(x: torch.Tensor) -> torch.Tensor:
    """Evaluate novel discrimination test function from Owen et al.

    The threshold is roughly parabolic with context, and the slope
    varies with the threshold. Adding to the difficulty is the fact
    that the function is minimized at f=0 (or p=0.5), corresponding
    to discrimination being at chance at zero stimulus intensity.

    Args:
        x (torch.Tensor): Points at which to evaluate.

    Returns:
        torch.Tensor: Value of function at these points.
    """
    freq = x[..., 0]
    amp = x[..., 1]
    context = 2 * (0.05 + 0.4 * (-1 + 0.2 * freq) ** 2 * freq**2)
    return 2 * (amp + 1) / context


def novel_detection_testfun(x: torch.Tensor) -> torch.Tensor:
    """Evaluate novel detection test function from Owen et al.

    The threshold is roughly parabolic with context, and the slope
    varies with the threshold.

    Args:
        x (torch.Tensor): Points at which to evaluate.

    Returns:
         torch.Tensor: Value of function at these points.
    """
    freq = x[..., 0]
    amp = x[..., 1]
    context = 2 * (0.05 + 0.4 * (-1 + 0.2 * freq) ** 2 * freq**2)
    return 4 * (amp + 1) / context - 4


def discrim_highdim(x: torch.Tensor) -> torch.Tensor:
    amp = x[..., 0]
    freq = x[..., 1]
    vscale = x[..., 2]
    vshift = x[..., 3]
    variance = x[..., 4]
    asym = x[..., 5]
    phase = x[..., 6]
    period = x[..., 7]

    context = (
        -0.5 * vscale * torch.cos(period * 0.6 * torch.pi * freq + phase)
        + vscale / 2
        + vshift
    ) * (
        -1 * asym * torch.sin(period * 0.6 * torch.pi * 0.5 * freq + phase) + (2 - asym)
    ) - 1

    z = (amp - context) / (variance + variance * (1 + context))
    normal_dist = torch.distributions.Normal(0, 1)
    p = normal_dist.cdf(z)
    p = (1 - 0.5) * p + 0.5
    p = torch.clamp(p, 0.5, 1 - 1e-5)

    return normal_dist.icdf(p)


def modified_hartmann6(X: torch.Tensor) -> torch.Tensor:
    """
    The modified Hartmann6 function used in Lyu et al.
    """
    C = torch.tensor([0.2, 0.22, 0.28, 0.3], dtype=torch.float64)
    a_t = torch.tensor(
        [
            [8, 3, 10, 3.5, 1.7, 6],
            [0.5, 8, 10, 1.0, 6, 9],
            [3, 3.5, 1.7, 8, 10, 6],
            [10, 6, 0.5, 8, 1.0, 9],
        ],
        dtype=torch.float64,
    )

    p_t = 10 ** (-4) * torch.tensor(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ],
        dtype=torch.float64,
    )

    y = torch.tensor(0.0, dtype=torch.float64)
    for i, C_i in enumerate(C):
        t = torch.tensor(0.0, dtype=torch.float64)
        for j in range(6):
            t += a_t[i, j] * ((X[j] - p_t[i, j]) ** 2)
        y += C_i * torch.exp(-t)
    return -10 * (y - 0.1)


def f_1d(x: torch.Tensor, mu: float = 0) -> torch.Tensor:
    """
    latent is just a gaussian bump at mu
    """
    return torch.exp(-((x - mu) ** 2))


def f_2d(x: torch.Tensor) -> torch.Tensor:
    """
    a gaussian bump at 0, 0
    """
    return torch.exp(-torch.norm(x, dim=-1))


def new_novel_det_params(
    freq: torch.Tensor, scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the loc and scale params for 2D synthetic novel_det(frequency) function
        Keyword arguments:
    freq -- 1D tensor of frequencies whose thresholds to return
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    """
    locs = 0.66 * torch.pow(0.8 * freq * (0.2 * freq - 1), 2) + 0.05
    scale = 2 * locs / (3 * scale_factor)
    loc = -1 + 2 * locs
    return loc, scale


def target_new_novel_det(
    freq: torch.Tensor, scale_factor: float = 1.0, target: float = 0.75
) -> torch.Tensor:
    """Get the target (i.e. threshold) for 2D synthetic novel_det(frequency) function
        Keyword arguments:
    freq -- 1D tensor of frequencies whose thresholds to return
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    target -- target threshold
    """
    locs, scale = new_novel_det_params(freq, scale_factor)
    normal_dist = torch.distributions.Normal(locs, scale)
    return normal_dist.icdf(torch.tensor(target))


def new_novel_det(x: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
    """Get the cdf for 2D synthetic novel_det(frequency) function
        Keyword arguments:
    x -- tensor of shape (n,2) of locations to sample;
         x[...,0] is frequency from -1 to 1; x[...,1] is intensity from -1 to 1
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    """
    freq = x[..., 0]
    locs, scale = new_novel_det_params(freq, scale_factor)
    return (x[..., 1] - locs) / scale


def cdf_new_novel_det(x: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
    """Get the cdf for 2D synthetic novel_det(frequency) function
        Keyword arguments:
    x -- tensor of shape (n,2) of locations to sample;
         x[...,0] is frequency from -1 to 1; x[...,1] is intensity from -1 to 1
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    """
    z = new_novel_det(x, scale_factor)
    normal_dist = torch.distributions.Normal(0, 1)  # Standard normal distribution
    return normal_dist.cdf(z)


def new_novel_det_channels_params(
    channel: torch.Tensor,
    scale_factor: float = 1.0,
    wave_freq: float = 1,
    target: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the target parameters for 2D synthetic novel_det(channel) function
        Keyword arguments:
    channel -- 1D tensor of channel locations whose thresholds to return
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    wave_freq -- frequency of location waveform on [-1,1]
    target -- target threshold
    """
    locs = -0.3 * torch.sin(5 * wave_freq * (channel - 1 / 6) / torch.pi) ** 2 - 0.5
    scale = (
        1
        / (10 * scale_factor)
        * (0.75 + 0.25 * torch.cos(10 * (0.3 + channel) / torch.pi))
    )
    return locs, scale


def target_new_novel_det_channels(
    channel: torch.Tensor,
    scale_factor: float = 1.0,
    wave_freq: float = 1,
    target: float = 0.75,
) -> torch.Tensor:
    """Get the target (i.e. threshold) for 2D synthetic novel_det(channel) function
        Keyword arguments:
    channel -- 1D tensor of channel locations whose thresholds to return
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    wave_freq -- frequency of location waveform on [-1,1]
    target -- target threshold
    """
    locs, scale = new_novel_det_channels_params(
        channel, scale_factor, wave_freq, target
    )
    normal_dist = torch.distributions.Normal(locs, scale)
    return normal_dist.icdf(torch.tensor(target))


def new_novel_det_channels(
    x: torch.Tensor,
    channel: torch.Tensor,
    scale_factor: float = 1.0,
    wave_freq: float = 1,
    target: float = 0.75,
) -> torch.Tensor:
    """Get the 2D synthetic novel_det(channel) function
        Keyword arguments:
    x -- tensor of shape (n,2) of locations to sample;
         x[...,0] is channel from -1 to 1; x[...,1] is intensity from -1 to 1
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    wave_freq -- frequency of location waveform on [-1,1]
    """
    locs, scale = new_novel_det_channels_params(
        channel, scale_factor, wave_freq, target
    )
    return (x[..., 1] - locs) / scale


def cdf_new_novel_det_channels(
    x: torch.Tensor,
    channel: torch.Tensor,
    scale_factor: float = 1.0,
    wave_freq: float = 1,
    target: float = 0.75,
) -> torch.Tensor:
    """Get the cdf for 2D synthetic novel_det(channel) function
        Keyword arguments:
    x -- tensor of shape (n,2) of locations to sample;
         x[...,0] is channel from -1 to 1; x[...,1] is intensity from -1 to 1
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    wave_freq -- frequency of location waveform on [-1,1]
    """
    z = new_novel_det_channels(x, channel, scale_factor, wave_freq, target)
    normal_dist = torch.distributions.Normal(0, 1)  # Standard normal distribution
    return normal_dist.cdf(z)


def new_novel_det_3D_params(
    x: torch.Tensor, scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    freq = x[..., 0]
    chan = x[..., 1]
    locs_freq = -0.32 + 2 * (0.66 * torch.pow(0.8 * freq * (0.2 * freq - 1), 2) + 0.05)
    locs = (
        0.7 * ((-0.35 * torch.sin(5 * (chan - 1 / 6) / torch.pi) ** 2) - 0.5)
        + 0.9 * locs_freq
    )
    scale = 0.3 * locs / (3 * scale_factor) * 1 / (10 * scale_factor) + 0.15 * (
        0.75 + 0.25 * torch.cos(10 * (0.6 + chan) / torch.pi)
    )
    return locs, scale


def new_novel_det_3D(x: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Get the synthetic 3D novel_det function over freqs, channels, and amplitudes.
    """
    locs, scale = new_novel_det_3D_params(x, scale_factor)
    return (x[..., 2] - locs) / scale


def cdf_new_novel_det_3D(x: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Get the cdf for 3D synthetic novel_det function

    x -- tensor of shape (n,3) of locations to sample
         x[...,0] is frequency, x[...,1] is channel, x[...,2] is intensity

    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    """
    z = new_novel_det_3D(x, scale_factor)
    normal_dist = torch.distributions.Normal(0, 1)  # Standard normal distribution
    return normal_dist.cdf(z)


def target_new_novel_det_3D(
    x: torch.Tensor, scale_factor: float = 1.0, target: float = 0.75
) -> torch.Tensor:
    """
    Get target for 3D synthetic novel_det function at location x

    x -- tensor of shape (n,2) of locations to sample
         x[...,0] is frequency, x[...,1] is channel,

    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    target -- target threshold
    """
    locs, scale = new_novel_det_3D_params(x, scale_factor)
    normal_dist = torch.distributions.Normal(locs, scale)
    return normal_dist.icdf(torch.tensor(target))


def f_pairwise(f: Callable, x: torch.Tensor, noise_scale: float = 1) -> torch.Tensor:
    normal_dist = torch.distributions.Normal(0, 1)
    return normal_dist.cdf(
        (f(x[..., 1]) - f(x[..., 0])) / (noise_scale * torch.sqrt(torch.tensor(2.0)))
    )
