#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import norm

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


def make_songetal_threshfun(x: np.ndarray, y: np.ndarray) -> Callable[[float], float]:
    """Generate a synthetic threshold function by interpolation of real data.

    Real data is from Dubno et al. 2013, and procedure follows Song et al. 2017, 2018.
    See make_songetal_testfun for more detail.

    Args:
        x (np.ndarray): Frequency
        y (np.ndarray): Threshold

    Returns:
        Callable[[float], float]: Function that interpolates the given
            frequencies and thresholds and returns threshold as a function
            of frequency.
    """
    f_interp = CubicSpline(x, y, extrapolate=False)
    f_extrap = interp1d(x, y, fill_value="extrapolate")

    def f_combo(x):
        # interpolate first
        interpolated = f_interp(x)
        # whatever is nan needs extrapolating
        interpolated[np.isnan(interpolated)] = f_extrap(x[np.isnan(interpolated)])
        return interpolated

    return f_combo


def make_songetal_testfun(
    phenotype: str = "Metabolic", beta: float = 1
) -> Callable[[np.ndarray, bool], np.ndarray]:
    """Make an audiometric test function following Song et al. 2017.

    To do so,we first compute a threshold by interpolation/extrapolation
    from real data, then assume a linear psychometric function in intensity
    with slope beta.

    Args:
        phenotype (str, optional): Audiometric phenotype from Dubno et al. 2013.
            Specifically, one of "Metabolic", "Sensory", "Metabolic+Sensory",
            or "Older-normal". Defaults to "Metabolic".
        beta (float, optional): Psychometric function slope. Defaults to 1.

    Returns:
        Callable[[np.ndarray, bool], np.ndarray]: A test function taking a [b x 2] array of points and returning the psychometric function value at those points.

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
    x = dubno_data[dubno_data.phenotype == phenotype].freq.values
    y = dubno_data[dubno_data.phenotype == phenotype].thresh.values
    # first, make the threshold fun
    threshfun = make_songetal_threshfun(x, y)

    # now make it into a test function
    def song_testfun(x, cdf=False):
        logfreq = x[..., 0]
        intensity = x[..., 1]
        thresh = threshfun(2**logfreq)
        return (
            norm.cdf((intensity - thresh) / beta)
            if cdf
            else (intensity - thresh) / beta
        )

    return song_testfun


def novel_discrimination_testfun(x: np.ndarray) -> np.ndarray:
    """Evaluate novel discrimination test function from Owen et al.

    The threshold is roughly parabolic with context, and the slope
    varies with the threshold. Adding to the difficulty is the fact
    that the function is minimized at f=0 (or p=0.5), corresponding
    to discrimination being at chance at zero stimulus intensity.

    Args:
        x (np.ndarray): Points at which to evaluate.

    Returns:
        np.ndarray: Value of function at these points.
    """
    freq = x[..., 0]
    amp = x[..., 1]
    context = 2 * (0.05 + 0.4 * (-1 + 0.2 * freq) ** 2 * freq**2)
    return 2 * (amp + 1) / context


def novel_detection_testfun(x: np.ndarray) -> np.ndarray:
    """Evaluate novel detection test function from Owen et al.

    The threshold is roughly parabolic with context, and the slope
    varies with the threshold.

    Args:
        x (np.ndarray): Points at which to evaluate.

    Returns:
        np.ndarray: Value of function at these points.
    """
    freq = x[..., 0]
    amp = x[..., 1]
    context = 2 * (0.05 + 0.4 * (-1 + 0.2 * freq) ** 2 * freq**2)
    return 4 * (amp + 1) / context - 4


def discrim_highdim(x: np.ndarray) -> np.ndarray:
    amp = x[..., 0]
    freq = x[..., 1]
    vscale = x[..., 2]
    vshift = x[..., 3]
    variance = x[..., 4]
    asym = x[..., 5]
    phase = x[..., 6]
    period = x[..., 7]

    context = (
        -0.5 * vscale * np.cos(period * 0.6 * math.pi * freq + phase)
        + vscale / 2
        + vshift
    ) * (
        -1 * asym * np.sin(period * 0.6 * math.pi * 0.5 * freq + phase) + (2 - asym)
    ) - 1
    z = (amp - context) / (variance + variance * (1 + context))
    p = norm.cdf(z)
    p = (1 - 0.5) * p + 0.5  # Floor at p=0.5
    p = np.clip(p, 0.5, 1 - 1e-5)  # clip so that norm.ppf doesn't go to inf
    return norm.ppf(p)


def modified_hartmann6(X):
    """
    The modified Hartmann6 function used in Lyu et al.
    """
    C = np.r_[0.2, 0.22, 0.28, 0.3]
    a_t = np.c_[
        [8, 3, 10, 3.5, 1.7, 6],
        [0.5, 8, 10, 1.0, 6, 9],
        [3, 3.5, 1.7, 8, 10, 6],
        [10, 6, 0.5, 8, 1.0, 9],
    ].T

    p_t = (
        10 ** (-4)
        * np.c_[
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ].T
    )

    y = 0.0
    for i, C_i in enumerate(C):
        t = 0
        for j in range(6):
            t += a_t[i, j] * ((X[j] - p_t[i, j]) ** 2)
        y += C_i * np.exp(-t)
    return -10 * (float(y) - 0.1)
