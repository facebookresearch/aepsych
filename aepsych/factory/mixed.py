#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Any, Literal

import botorch
import gpytorch
import torch
from aepsych.config import Config
from aepsych.factory.default import DefaultMeanCovarFactory
from aepsych.factory.utils import temporary_attributes


class MixedMeanCovarFactory(DefaultMeanCovarFactory):
    def __init__(
        self,
        dim: int,
        discrete_params: dict[int, int],
        stimuli_per_trial: int = 1,
        discrete_param_ranks: dict[int, int] | None = None,
        discrete_kernel: Literal["index", "categorical"] = "categorical",
        zero_mean: bool = False,
        target: float | None = None,
        cov_kernel: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel,
        active_dims: list[int] | None = None,
        lengthscale_prior: Literal["invgamma", "gamma", "lognormal"] | None = None,
        ls_loc: torch.Tensor | float | None = None,
        ls_scale: torch.Tensor | float | None = None,
        fixed_kernel_amplitude: bool | None = None,
        outputscale_prior: Literal["box", "gamma"] = "box",
    ) -> None:
        """Factory that makes mean and covariance functions for generic GPs.
        After initialization, copies of the mean and covariance functions can be made with
        `get_mean` and `get_covar`.

        Args:
            dim (int, optional): Dimensionality of the parameter space. Must be provided.
            stimuli_per_trial (int): Number of stimuli per trial. Defaults to 1.
            zero_mean (bool, optional): Whether to use zero for the mean module. Defaults to False.
            target (float, optional): Target for the mean module. Defaults to None.
            cov_kernel (gpytorch.kernels.Kernel, optional): Covariance kernel to use. Defaults to RBF
                kernel.
            active_dims (list[int], optional): List of dimensions to use in the covariance function. Defaults to None,
                which uses all dimensions.
            lengthscale_prior (Literal["invgamma", "gamma", "lognormal"], optional): Prior to use for
                lengthscale. Defaults to "lognormal" if stimuli_per_trial == 1, else "gamma".
            ls_loc (torch.Tensor | float, optional): Location parameter for lengthscale prior.
                Defaults to sqrt(2.0).
            ls_scale (torch.Tensor | float, optional): Scale parameter for lengthscale prior.
                Defaults to sqrt(3.0).
            fixed_kernel_amplitude (bool, optional): Whether to allow the covariance kernel to scale.
                Defaults to True if stimuli_per_trial == 1, else False.
            outputscale_prior (Literal["box", "gamma"], optional): Prior to use to scale the covariance kernel.
                Defaults to "box".
        """
        discrete_param_ranks = discrete_param_ranks or discrete_params.copy()

        # Check if the keys in both dictionaries match
        if set(discrete_params.keys()) != set(discrete_param_ranks.keys()):
            raise ValueError("discrete parameter indices and ranks should match")

        if discrete_kernel not in ("index", "categorical"):
            raise ValueError(
                "only index or categorical kernels supported for discrete kernel"
            )

        self.discrete_params = discrete_params
        self.discrete_param_ranks = discrete_param_ranks or discrete_params.copy()
        self.discrete_kernel = discrete_kernel
        self.zero_mean = zero_mean
        self.target = target
        self.cov_kernel = cov_kernel
        self.active_dims = active_dims
        self.lengthscale_prior = lengthscale_prior
        self.ls_loc = ls_loc
        self.ls_scale = ls_scale
        self.fixed_kernel_amplitude = fixed_kernel_amplitude
        self.outputscale_prior = outputscale_prior

        super().__init__(dim, stimuli_per_trial)

    def _make_covar_module(self) -> gpytorch.kernels.Kernel:
        # Make covariance module
        cont_dims = self.active_dims or list(range(self.dim))
        cont_dims = [idx for idx in cont_dims if idx not in self.discrete_params.keys()]
        with temporary_attributes(
            self, dim=len(cont_dims), fixed_kernel_amplitude=True, active_dims=cont_dims
        ):
            cont_kernel = super()._make_covar_module()

        if self.discrete_kernel == "index":
            discrete_kernels = []
            for idx in self.discrete_params.keys():
                discrete_kernels.append(
                    gpytorch.kernels.IndexKernel(
                        num_tasks=self.discrete_params[idx],
                        rank=self.discrete_param_ranks[idx],
                        active_dims=(idx,),
                        ard_num_dims=1,
                        prior=gpytorch.priors.LKJCovariancePrior(
                            n=self.discrete_param_ranks[idx],
                            eta=1.5,
                            sd_prior=gpytorch.priors.GammaPrior(1.0, 0.15),
                        ),
                    )
                )
            add_kernel = gpytorch.kernels.AdditiveKernel(
                deepcopy(cont_kernel), *deepcopy(discrete_kernels)
            )
            prod_kernel = gpytorch.kernels.ProductKernel(
                deepcopy(cont_kernel), *deepcopy(discrete_kernels)
            )
            return add_kernel * prod_kernel
        elif self.discrete_kernel == "categorical":
            constraint = gpytorch.constraints.GreaterThan(lower_bound=1e-4)
            discrete_kernel = botorch.models.kernels.CategoricalKernel(
                active_dims=tuple(self.discrete_params.keys()),
                ard_num_dims=len(self.discrete_params),
                lengthscale_constraint=constraint,
            )

            if not self.fixed_kernel_amplitude:
                discrete_kernel = gpytorch.kernels.ScaleKernel(discrete_kernel)
                cont_kernel = gpytorch.kernels.ScaleKernel(cont_kernel)

            add_kernel = deepcopy(cont_kernel) + deepcopy(discrete_kernel)
            prod_kernel = deepcopy(cont_kernel) * deepcopy(discrete_kernel)

            return add_kernel * prod_kernel
        else:
            raise ValueError("discrete kernel must be index or categorical")

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get configuration options for the MeanCovarFactory.

        Args:
            config (Config): Config object to find options in.
            name (str, optional): Name of the factory. Defaults to the class name.
            options (dict, optional): Options to start with. Defaults to None.

        Returns:
            dict[str, Any]: Options to use to initialize the factory.
        """
        name = name or cls.__name__
        options = super().get_config_options(config, name, options)

        # Figure out discrete parameters
        par_names = config.getlist("common", "parnames", element_type=str)
        discrete_params = {}
        discrete_ranks = {}
        for i, par_name in enumerate(par_names):
            if config.get(par_name, "par_type") == "categorical":
                discrete_params[i] = len(
                    config.getlist(par_name, "choices", element_type=str)
                )
                discrete_ranks[i] = config.getint(
                    par_name, "rank", fallback=discrete_params[i]
                )

        if len(discrete_params) == 0:
            raise ValueError("No categorical parameters found")

        options["discrete_params"] = discrete_params
        options["discrete_param_ranks"] = discrete_ranks

        return options
