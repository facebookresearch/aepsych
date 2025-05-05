#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Literal

import gpytorch
import torch
from aepsych.config import Config
from aepsych.factory.default import (
    _get_default_cov_function,
    _get_default_mean_function,
    default_mean_covar_factory,
    DefaultMeanCovarFactory,
)
from aepsych.factory.utils import temporary_attributes
from aepsych.kernels.pairwisekernel import PairwiseKernel


class PairwiseMeanCovarFactory(DefaultMeanCovarFactory):
    def __init__(
        self,
        dim: int,
        stimuli_per_trial: int = 1,
        shared_dims: list[int] | None = None,
        zero_mean: bool = True,
        target: float | None = None,
        cov_kernel: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel,
        lengthscale_prior: Literal["invgamma", "gamma", "lognormal"] | None = None,
        ls_loc: torch.Tensor | float | None = None,
        ls_scale: torch.Tensor | float | None = None,
        fixed_kernel_amplitude: bool | None = None,
        outputscale_prior: Literal["box", "gamma"] = "box",
    ) -> None:
        if stimuli_per_trial != 1:
            raise ValueError(
                "Pairwise kernels can only be used with 1 stimuli per trial as the points "
                f"are returned as a single flat configuration, got {stimuli_per_trial=}"
            )

        if dim < 2:
            raise ValueError(
                f"Pairwise kernels require at least 2 dimensions, got {dim=}"
            )

        if shared_dims is not None:
            self.shared_dims = [int(d) for d in shared_dims]
            for shared_dim in self.shared_dims:
                if shared_dim >= dim:
                    raise ValueError(
                        f"shared dim is out of bounds: {shared_dim=}, {dim=}"
                    )
        else:
            self.shared_dims = []

        if len(self.shared_dims) > dim:
            raise ValueError(
                "There are more shared dimensions than the number of dimensions!"
            )

        super().__init__(
            dim=dim,
            stimuli_per_trial=stimuli_per_trial,
            zero_mean=zero_mean,
            target=target,
            cov_kernel=cov_kernel,
            lengthscale_prior=lengthscale_prior,
            ls_loc=ls_loc,
            ls_scale=ls_scale,
            fixed_kernel_amplitude=fixed_kernel_amplitude,
            outputscale_prior=outputscale_prior,
        )

    def _make_covar_module(self) -> gpytorch.kernels.Kernel:
        # Make the covariance module
        active_dims = [i for i in range(self.dim) if i not in self.shared_dims]

        if len(active_dims) % 2 != 0:
            raise ValueError(
                f"Number of active dims must be even, got {len(active_dims)=}, {self.dim=}, {self.shared_dims=}"
            )

        # Temporarily modify attributes to make base pair covariance module
        with temporary_attributes(self, dim=len(active_dims) // 2, stimuli_per_trial=1):
            base_cov = super()._make_covar_module()

        if len(self.shared_dims) == 0:
            return PairwiseKernel(base_cov)

        else:  # Some paired dims
            # Again temporary attributes
            with temporary_attributes(
                self,
                dim=len(self.shared_dims),
                active_dims=self.shared_dims,
                stimuli_per_trial=1,
            ):
                shared_cov = super()._make_covar_module()

            return PairwiseKernel(base_cov, active_dims=active_dims) * shared_cov


def pairwise_mean_covar_factory(
    config: Config,
    dim: int | None = None,
    stimuli_per_trial: int = 1,
) -> tuple[gpytorch.means.ConstantMean, gpytorch.kernels.ScaleKernel]:
    """Creates a mean and covariance function for pairwise GPs.

    Args:
        config (Config): Config object containing bounds.
        dim (int, optional): Dimensionality of the parameter space. Unused; here for API consistency with the default factory.
        stimuli_per_trial (int): Number of stimuli per trial. Because this factory is intended to be used with GPClassificationModel, this must actually be 1.
    Returns:
        tuple[gpytorch.means.ConstantMean, gpytorch.kernels.ScaleKernel]: A tuple containing
        the mean function (ConstantMean) and the covariance function (ScaleKernel)."""

    warnings.warn(
        "pairwise_mean_covar_factory is deprecated, use the PairwiseMeanCovarFactory class instead!",
        DeprecationWarning,
        stacklevel=2,
    )

    assert (
        stimuli_per_trial == 1
    ), f"pairwise_mean_covar_factory must have stimuli_per_trial == 1, but {stimuli_per_trial} was passed instead!"
    lb = config.gettensor("common", "lb")
    ub = config.gettensor("common", "ub")
    assert lb.shape[0] == ub.shape[0], "bounds shape mismatch!"
    assert lb.shape[0] >= 2, "PairwiseKernel requires at least 2 dimensions!"
    config_dim = lb.shape[0]

    shared_dims: list[int] | None = config.getlist(
        "pairwise_mean_covar_factory", "shared_dims", fallback=None
    )
    if shared_dims is not None:
        shared_dims = [int(d) for d in shared_dims]
        assert len(shared_dims) < config_dim, "length of shared_dims must be < dim!"
        for dim in shared_dims:
            assert dim < len(shared_dims)
    else:
        shared_dims = []

    base_mean_covar_factory = config.getobj(
        "pairwise_mean_covar_factory",
        "base_mean_covar_factory",
        fallback=default_mean_covar_factory,
    )

    if base_mean_covar_factory is not default_mean_covar_factory:
        raise NotImplementedError(
            "Only default_mean_covar_factory is supported for the base factor of pairwise_mean_covar_factory right now!"
        )

    zero_mean = config.getboolean(
        "pairwise_mean_covar_factory", "zero_mean", fallback=True
    )

    if len(shared_dims) > 0:
        active_dims = [i for i in range(config_dim) if i not in shared_dims]
        assert (
            len(active_dims) % 2 == 0
        ), "dimensionality of non-shared dims must be even!"
        mean = _get_default_mean_function(config, zero_mean)
        cov1 = _get_default_cov_function(
            config, len(active_dims) // 2, stimuli_per_trial=1
        )

        cov2 = _get_default_cov_function(
            config, len(shared_dims), active_dims=shared_dims, stimuli_per_trial=1
        )

        covar = PairwiseKernel(cov1, active_dims=active_dims) * cov2

    else:
        assert config_dim % 2 == 0, "dimensionality must be even!"
        mean = _get_default_mean_function(config, zero_mean)
        cov = _get_default_cov_function(config, config_dim // 2, stimuli_per_trial=1)
        covar = PairwiseKernel(cov)

    return mean, covar
