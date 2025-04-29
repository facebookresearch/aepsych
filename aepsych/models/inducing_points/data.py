#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
from aepsych.models.inducing_points.base import BaseAllocator, EMPTY_SIZE


class DataAllocator(BaseAllocator):
    def __init__(
        self,
        dim: int,
    ) -> None:
        """Initialize the DataAllocator. This allocator simply returns the input
        data to use as the inducing points.

        Args:
            dim (int): Dimensionality of the search space.
        """
        super().__init__(dim=dim)

    def allocate_inducing_points(
        self,
        inputs: torch.Tensor | None = None,
        covar_module: torch.nn.Module | None = None,
        num_inducing: int = 100,
        input_batch_shape: torch.Size = EMPTY_SIZE,
    ) -> torch.Tensor:
        """Allocate inducing points by returning the inputs as the inducing points.

        Args:
            inputs (torch.Tensor): Input tensor, cloned and returned as inducing points.
            covar_module (torch.nn.Module, optional): Kernel covariance module; included for API compatibility, but not used here.
            num_inducing (int, optional): The number of inducing points to generate. This parameter is ignored by DataAllocator,
                which always returns all input points.
            input_batch_shape (torch.Size, optional): Batch shape; included for API compatibility, but not used here.

        Returns:
            torch.Tensor: The input data as inducing points.
        """
        if inputs is None:  # Dummy points
            return self._allocate_dummy_points(num_inducing=num_inducing)

        if num_inducing < inputs.shape[0]:
            warnings.warn(
                f"DataAllocator ignores num_inducing={num_inducing} and returns all input points.",
                UserWarning,
                stacklevel=2,
            )

        self.last_allocator_used = self.__class__
        return inputs.clone().detach()
