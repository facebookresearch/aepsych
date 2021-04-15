#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
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


def make_songetal_threshfun(x, y):
    """
    makes a function that generates a threshold function as in Song
    et al.
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


def make_songetal_testfun(phenotype="Metabolic", beta=1):
    """
    Makes a test function as used by:
    Song, X. D., Garnett, R., & Barbour, D. L. (2017).
    Psychometric function estimation by probabilistic classification.
    The Journal of the Acoustical Society of America, 141(4), 2513–2525.
    https://doi.org/10.1121/1.4979594
    """
    valid_phenotypes = ["Metabolic", "Sensory", "Metabolic+Sensory", "Older-normal"]
    assert phenotype in valid_phenotypes, f"Phenotype must be one of {valid_phenotypes}"
    x = dubno_data[dubno_data.phenotype == phenotype].freq
    y = dubno_data[dubno_data.phenotype == phenotype].thresh
    # first, make the threshold fun
    threshfun = make_songetal_threshfun(x, y)

    # now make it into a test function
    def song_testfun(x, cdf=False):
        logfreq = x[..., 0]
        intensity = x[..., 1]
        thresh = threshfun(2 ** logfreq)
        return (
            norm.cdf((intensity - thresh) / beta)
            if cdf
            else (intensity - thresh) / beta
        )

    return song_testfun


def novel_discrimination_testfun(x):
    freq = x[..., 0]
    amp = x[..., 1]
    context = 2 * (0.05 + 0.4 * (-1 + 0.2 * freq) ** 2 * freq ** 2)
    return 2 * (amp + 1) / context


def novel_detection_testfun(x):
    freq = x[..., 0]
    amp = x[..., 1]
    context = 2 * (0.05 + 0.4 * (-1 + 0.2 * freq) ** 2 * freq ** 2)
    return 4 * (amp + 1) / context - 4
