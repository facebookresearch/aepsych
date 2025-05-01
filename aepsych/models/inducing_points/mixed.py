#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import cached_property
from typing import Any

import torch
from aepsych.config import Config
from aepsych.models.inducing_points.base import BaseAllocator, EMPTY_SIZE


class MixedBaseAllocator(BaseAllocator):
    def __init__(
        self,
        dim: int,
        categorical_params: dict[int, int],
        continuous_allocator: type[BaseAllocator],
        **kwargs: Any,
    ) -> None:
        """Base class for mixed allocators. This class splits the input into
        continuous and categorical parts and then allocates inducing points for
        the categorical parts using the continuous alloctor. The different sub-
        classes are largely different only in their allocate_inducing_points
        method.

        Args:
            dim (int): Dimensionality of the mixed search space.
            categorical_params (dict[int, int]): Dictionary specifying which parameters are
                categorical and how many options they have.
            continuous_allocator (type[BaseAllocator]): The type of allocator to use for the
                continuous parameters. This will be initialized with the additional **kwargs.
            **kwargs: Keyword arguments to pass to the continuous_allocator to initialize it.
        """
        super().__init__(dim=dim)
        self.categorical_params = categorical_params
        self.categorical_idxs = sorted(categorical_params.keys())
        self.continuous_idxs = sorted(set(range(dim)) - set(self.categorical_idxs))

        # Initialize the continuous allocator
        self.continuous_allocator = continuous_allocator(
            dim=len(self.continuous_idxs), **kwargs
        )

        # Check if the continuous allocator produces the right shape
        dummy = self.continuous_allocator.allocate_inducing_points(inputs=None)
        if dummy.shape[1] != len(self.continuous_idxs):
            raise ValueError(
                "The continuous allocator does not produce the right shape. "
                f"Got {dummy.shape[1]} and expected {len(self.continuous_idxs)}. "
                "The kwargs for the continuous allocator should be chosen as if only "
                "the continuous parameters were present."
            )

    @cached_property
    def categorical_points(self) -> torch.Tensor:
        """Return a tensor of all categorication configurations given the
        categorical  parameters. This is cached.

        Returns:
            torch.Tensor: Tensor of all categorical configurations.
        """
        points = torch.cartesian_prod(
            *[torch.arange(self.categorical_params[i]) for i in self.categorical_idxs]
        )

        if len(points.shape) == 1:
            points = points.unsqueeze(1)

        return points

    def _split_inputs(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the inputs into continuous and categorical parts.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Continuous and categorical parts of the input.
        """
        return inputs[:, self.continuous_idxs], inputs[:, self.categorical_idxs]

    def _combine_inducing_points(
        self, continuous_induc: torch.Tensor, categorical_induc: torch.Tensor
    ) -> torch.Tensor:
        """Combine continuous and categorical inducing points into a single tensor.

        Args:
            continuous_induc (torch.Tensor): Continuous inducing points.
            categorical_induc (torch.Tensor): Categorical inducing points.

        Returns:
            torch.Tensor: Combined inducing points.
        """
        categorical_induc = categorical_induc.to(continuous_induc)

        # Make a dummy tensor to fill in
        inducing_points = torch.empty((continuous_induc.shape[0], self.dim))
        inducing_points = inducing_points.to(continuous_induc)
        inducing_points[:, self.continuous_idxs] = continuous_induc
        inducing_points[:, self.categorical_idxs] = categorical_induc

        return inducing_points

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get configuration options for the categorical allocator.

        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the allocator, defaults to None. Ignored.
            options (dict[str, Any], optional): Additional options, defaults to None.

        Returns:
            dict[str, Any]: Configuration options for the CategoricalAllocator.
        """
        options = super().get_config_options(config, name, options)

        par_names = config.getlist("common", "parnames", element_type=str)
        categorical_params: dict[int, int] = {}
        for i, par_name in enumerate(par_names):
            if config.get(par_name, "par_type") == "categorical":
                categorical_params[i] = len(config.getlist(par_name, "options"))

        options["categorical_params"] = categorical_params

        # TODO: Check if we need extra logic to initialize the continuous allocator

        return options


class SubsetMixedAllocator(MixedBaseAllocator):
    """Inducing point allocator for mixed input models that places continuous inducing points
    on a random subset of the categorical indices.
    """

    def allocate_inducing_points(
        self,
        inputs: torch.Tensor | None = None,
        covar_module: torch.nn.Module | None = None,
        num_inducing: int = 100,
        input_batch_shape: torch.Size = EMPTY_SIZE,
    ) -> torch.Tensor:
        """Allocate inducing points by placing continuous inducing points on a random subset
        of the categorical configurations.

        Args:
            inputs (torch.Tensor, optional): Input tensor containing both continuous and categorical parts.
            covar_module (torch.nn.Module, optional): Kernel covariance module.
            num_inducing (int, optional): The number of inducing points to generate. Defaults to 100.
            input_batch_shape (torch.Size, optional): Batch shape, defaults to an empty size.

        Returns:
            torch.Tensor: The allocated inducing points.
        """
        if inputs is None:
            return self._allocate_dummy_points(num_inducing)

        # Split inputs into continuous parts
        x_continuous = self._split_inputs(inputs)[0]

        # Create continuous inducing points
        continuous_induc = self.continuous_allocator.allocate_inducing_points(
            inputs=x_continuous,
            covar_module=covar_module,
            num_inducing=num_inducing,
            input_batch_shape=input_batch_shape,
        )

        # Generate all possible combinations of categorical parameters
        idx = torch.randint(0, self.categorical_points.shape[0], (num_inducing,))
        categorical_induc = self.categorical_points[idx].clone()

        # Combine continuous and categorical inducing points
        inducing_points = self._combine_inducing_points(
            continuous_induc=continuous_induc, categorical_induc=categorical_induc
        )

        self.last_allocator_used = self.__class__
        return inducing_points


class AllMixedAllocator(MixedBaseAllocator):
    """Inducing point allocator for mixed input models that places continuous inducing points
    for each permutation of the categorical indices. Probably doesn't scale very well so this
    should primarily be used for analysis.
    """

    def allocate_inducing_points(
        self,
        inputs: torch.Tensor | None = None,
        covar_module: torch.nn.Module | None = None,
        num_inducing: int = 100,
        input_batch_shape: torch.Size = EMPTY_SIZE,
    ) -> torch.Tensor:
        """Allocate inducing points by placing continuous inducing points for each
        permutation of the categorical indices.

        Args:
            inputs (torch.Tensor, optional): Input tensor containing both continuous and categorical parts.
            covar_module (torch.nn.Module, optional): Kernel covariance module.
            num_inducing (int, optional): Ignored as this allocator will generate an inducing point for each
                possible categorical configuration.
            input_batch_shape (torch.Size, optional): Batch shape, defaults to an empty size.

        Returns:
            torch.Tensor: The allocated inducing points.
        """
        if inputs is None:
            return self._allocate_dummy_points(num_inducing)

        # Split inputs into continuous parts
        x_continuous = self._split_inputs(inputs)[0]

        # Generate a continuous inducing point for each categorical configuration
        continuous_induc = self.continuous_allocator.allocate_inducing_points(
            inputs=x_continuous,
            covar_module=covar_module,
            num_inducing=self.categorical_points.shape[0],
            input_batch_shape=input_batch_shape,
        )

        if continuous_induc.shape[0] != self.categorical_points.shape[0]:
            raise ValueError(
                "The continuous allocator did not produce enough inducing points, this "
                "likely means the continuous allocator is not compatible with the AllMixedAllocator. "
                f"Got {continuous_induc.shape[0]} and {self.categorical_points.shape[0]}."
            )

        # Combine continuous and categorical inducing points
        inducing_points = self._combine_inducing_points(
            continuous_induc=continuous_induc,
            categorical_induc=self.categorical_points.clone(),
        )

        self.last_allocator_used = self.__class__
        return inducing_points
