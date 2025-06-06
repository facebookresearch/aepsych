#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from configparser import NoOptionError

import gpytorch
import torch
from aepsych.config import Config
from scipy.stats import norm

from .default import MeanCovarFactory
from .utils import (
    __default_invgamma_concentration,
    __default_invgamma_rate,
    DEFAULT_INVGAMMA_CONC,
    DEFAULT_INVGAMMA_RATE,
)


class SongMeanCovarFactory(MeanCovarFactory):
    def __init__(
        self,
        dim: int,
        stimuli_per_trial: int = 1,
        target: float = 0.75,
        stim_dim: int = -1,
        intensity_RBF: bool = False,
    ) -> None:
        self.target = target
        self.stim_dim = stim_dim
        self.intensity_RBF = intensity_RBF

        super().__init__(dim=dim, stimuli_per_trial=stimuli_per_trial)

    def _make_mean_module(self) -> gpytorch.means.Mean:
        # Make a mean module
        mean = gpytorch.means.ConstantMean()

        mean.constant.requires_grad_(False)
        mean.constant.copy_(torch.tensor(norm.ppf(self.target)))

        return mean

    def _make_covar_module(self) -> gpytorch.kernels.Kernel:
        # Make a covar module
        ls_prior = gpytorch.priors.GammaPrior(
            concentration=DEFAULT_INVGAMMA_CONC,
            rate=DEFAULT_INVGAMMA_RATE,
            transform=lambda x: 1 / x,
        )
        ls_prior_mode = ls_prior.rate / (ls_prior.concentration + 1)

        ls_constraint = gpytorch.constraints.GreaterThan(
            lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
        )

        context_dims = list(range(self.dim))
        # if intensity RBF is true, the intensity dimension
        # will have both the RBF and linear kernels
        if not self.intensity_RBF:
            intensity_dim = 1
            self.stim_dim = context_dims.pop(
                self.stim_dim
            )  # support relative stim dims
        else:
            intensity_dim = 0
            self.stim_dim = context_dims[self.stim_dim]

        # create the LinearKernel
        intensity_covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel(active_dims=self.stim_dim, ard_num_dims=1),
            outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
        )

        if self.dim == 1:
            # this can just be LinearKernel but for consistency of interface
            # we make it additive with one module
            if not self.intensity_RBF:
                return gpytorch.kernels.AdditiveKernel(intensity_covar)

            else:
                context_covar = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(
                        lengthscale_prior=ls_prior,
                        lengthscale_constraint=ls_constraint,
                        ard_num_dims=self.dim,
                        active_dims=context_dims,
                    ),
                    outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
                )
                return context_covar + intensity_covar
        else:
            context_covar = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale_prior=ls_prior,
                    lengthscale_constraint=ls_constraint,
                    ard_num_dims=self.dim - intensity_dim,
                    active_dims=context_dims,
                ),
                outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
            )
            return context_covar + intensity_covar


def song_mean_covar_factory(
    config: Config,
) -> tuple[gpytorch.means.ConstantMean, gpytorch.kernels.AdditiveKernel]:
    """
    Factory that makes kernels like Song et al. 2018:
    Linear in intensity dimension (assumed to be the last
    dimension), RBF in context dimensions, summed.

    Args:
        config (Config): Config object containing (at least) bounds and optionally
            LSE target.

    Returns:
        tuple[gpytorch.means.ConstantMean, gpytorch.kernels.AdditiveKernel]: Instantiated
            constant mean object and additive kernel object.
    """
    warnings.warn(
        "song_mean_covar_factory is deprecated, use the SongMeanCovarFactory class instead!",
        DeprecationWarning,
        stacklevel=2,
    )

    lb = config.gettensor("song_mean_covar_factory", "lb")
    ub = config.gettensor("song_mean_covar_factory", "ub")
    assert lb.shape[0] == ub.shape[0], "bounds shape mismatch!"
    dim = lb.shape[0]

    mean = gpytorch.means.ConstantMean()

    try:
        target = config.getfloat("song_mean_covar_factory", "target")
    except NoOptionError:
        target = 0.75
    mean.constant.requires_grad_(False)
    mean.constant.copy_(torch.tensor(norm.ppf(target)))

    ls_prior = gpytorch.priors.GammaPrior(
        concentration=__default_invgamma_concentration,
        rate=__default_invgamma_rate,
        transform=lambda x: 1 / x,
    )
    ls_prior_mode = ls_prior.rate / (ls_prior.concentration + 1)

    ls_constraint = gpytorch.constraints.GreaterThan(
        lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
    )

    stim_dim = config.getint("song_mean_covar_factory", "stim_dim", fallback=-1)
    context_dims = list(range(dim))
    # if intensity RBF is true, the intensity dimension
    # will have both the RBF and linear kernels
    intensity_RBF = config.getboolean(
        "song_mean_covar_factory", "intensity_RBF", fallback=False
    )
    if not intensity_RBF:
        intensity_dim = 1
        stim_dim = context_dims.pop(stim_dim)  # support relative stim dims
    else:
        intensity_dim = 0
        stim_dim = context_dims[stim_dim]

    # create the LinearKernel
    intensity_covar = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.LinearKernel(active_dims=stim_dim, ard_num_dims=1),
        outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
    )

    if dim == 1:
        # this can just be LinearKernel but for consistency of interface
        # we make it additive with one module
        if not intensity_RBF:
            return (
                mean,
                gpytorch.kernels.AdditiveKernel(intensity_covar),
            )
        else:
            context_covar = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale_prior=ls_prior,
                    lengthscale_constraint=ls_constraint,
                    ard_num_dims=dim,
                    active_dims=context_dims,
                ),
                outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
            )
            return mean, context_covar + intensity_covar
    else:
        context_covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint,
                ard_num_dims=dim - intensity_dim,
                active_dims=context_dims,
            ),
            outputscale_prior=gpytorch.priors.SmoothedBoxPrior(a=1, b=4),
        )
        return mean, context_covar + intensity_covar
