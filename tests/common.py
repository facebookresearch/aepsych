#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.stats import norm


def f_1d(x, mu=0):
    """
    latent is just a gaussian bump at mu
    """
    return np.exp(-((x - mu) ** 2))


def f_2d(x):
    """
    a gaussian bump at 0 , 0
    """
    return np.exp(-np.linalg.norm(x, axis=-1))


def new_novel_det_params(freq, scale_factor=1.0):
    """Get the loc and scale params for 2D synthetic novel_det(frequency) function
        Keyword arguments:
    freq -- 1D array of frequencies whose thresholds to return
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    target -- target threshold
    """
    locs = 0.66 * np.power(0.8 * freq * (0.2 * freq - 1), 2) + 0.05
    scale = 2 * locs / (3 * scale_factor)
    loc = -1 + 2 * locs
    return loc, scale


def target_new_novel_det(freq, scale_factor=1.0, target=0.75):
    """Get the target (i.e. threshold) for 2D synthetic novel_det(frequency) function
        Keyword arguments:
    freq -- 1D array of frequencies whose thresholds to return
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    target -- target threshold
    """
    locs, scale = new_novel_det_params(freq, scale_factor)
    return norm.ppf(target, loc=locs, scale=scale)


def new_novel_det(x, scale_factor=1.0):
    """Get the cdf for 2D synthetic novel_det(frequency) function
        Keyword arguments:
    x -- array of shape (n,2) of locations to sample;
         x[...,0] is frequency from -1 to 1; x[...,1] is intensity from -1 to 1
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    """
    freq = x[..., 0]
    locs, scale = new_novel_det_params(freq, scale_factor)
    return (x[..., 1] - locs) / scale


def cdf_new_novel_det(x, scale_factor=1.0):
    """Get the cdf for 2D synthetic novel_det(frequency) function
        Keyword arguments:
    x -- array of shape (n,2) of locations to sample;
         x[...,0] is frequency from -1 to 1; x[...,1] is intensity from -1 to 1
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    """
    return norm.cdf(new_novel_det(x, scale_factor))


def new_novel_det_channels_params(channel, scale_factor=1.0, wave_freq=1, target=0.75):
    """Get the target parameters for 2D synthetic novel_det(channel) function
        Keyword arguments:
    channel -- 1D array of channel locations whose thresholds to return
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    wave_freq -- frequency of location waveform on [-1,1]
    target -- target threshold
    """
    locs = -0.3 * np.sin(5 * wave_freq * (channel - 1 / 6) / np.pi) ** 2 - 0.5
    scale = (
        1 / (10 * scale_factor) * (0.75 + 0.25 * np.cos(10 * (0.3 + channel) / np.pi))
    )
    return locs, scale


def target_new_novel_det_channels(channel, scale_factor=1.0, wave_freq=1, target=0.75):
    """Get the target (i.e. threshold) for 2D synthetic novel_det(channel) function
        Keyword arguments:
    channel -- 1D array of channel locations whose thresholds to return
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    wave_freq -- frequency of location waveform on [-1,1]
    target -- target threshold
    """
    locs, scale = new_novel_det_channels_params(
        channel, scale_factor, wave_freq, target
    )
    return norm.ppf(target, loc=locs, scale=scale)


def new_novel_det_channels(x, channel, scale_factor=1.0, wave_freq=1, target=0.75):
    """Get the 2D synthetic novel_det(channel) function
        Keyword arguments:
    x -- array of shape (n,2) of locations to sample;
         x[...,0] is channel from -1 to 1; x[...,1] is intensity from -1 to 1
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    wave_freq -- frequency of location waveform on [-1,1]
    """
    locs, scale = new_novel_det_channels_params(
        channel, scale_factor, wave_freq, target
    )
    return (x - locs) / scale


def cdf_new_novel_det_channels(channel, scale_factor=1.0, wave_freq=1, target=0.75):
    """Get the cdf for 2D synthetic novel_det(channel) function
        Keyword arguments:
    x -- array of shape (n,2) of locations to sample;
         x[...,0] is channel from -1 to 1; x[...,1] is intensity from -1 to 1
    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    wave_freq -- frequency of location waveform on [-1,1]
    """
    return norm.cdf(new_novel_det_channels(channel, scale_factor, wave_freq, target))


def new_novel_det_3D_params(x, scale_factor=1.0):
    freq = x[..., 0]
    chan = x[..., 1]
    locs_freq = -0.32 + 2 * (0.66 * np.power(0.8 * freq * (0.2 * freq - 1), 2) + 0.05)
    locs = (
        0.7 * ((-0.35 * np.sin(5 * (chan - 1 / 6) / np.pi) ** 2) - 0.5)
        + 0.9 * locs_freq
    )
    scale = 0.3 * locs / (3 * scale_factor) * 1 / (10 * scale_factor) + 0.15 * (
        0.75 + 0.25 * np.cos(10 * (0.6 + chan) / np.pi)
    )
    return locs, scale


def new_novel_det_3D(x, scale_factor=1.0):
    """
    Get the synthetic 3D novel_det
    function over freqs,channels and amplitudes

    """
    locs, scale = new_novel_det_3D_params(x, scale_factor)
    return (x[..., 2] - locs) / scale


def cdf_new_novel_det_3D(x, scale_factor=1.0):
    """
    Get the cdf for 3D synthetic novel_det function

    x -- array of shape (n,3) of locations to sample
         x[...,0] is frequency, x[...,1] is channel, x[...,2] is intensity

    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    """
    return norm.cdf(new_novel_det_3D(x, scale_factor))


def target_new_novel_det_3D(x, scale_factor=1.0, target=0.75):
    """
    Get target for 3D synthetic novel_det function at location x

    x -- array of shape (n,2) of locations to sample
         x[...,0] is frequency, x[...,1] is channel,

    scale factor -- scale for the novel_det function, where higher is steeper/lower SD
    target -- target threshold

    """
    locs, scale = new_novel_det_3D_params(x, scale_factor)
    return norm.ppf(target, loc=locs, scale=scale)


def f_pairwise(f, x, noise_scale=1):
    return norm.cdf((f(x[..., 1]) - f(x[..., 0])) / (noise_scale * np.sqrt(2)))
