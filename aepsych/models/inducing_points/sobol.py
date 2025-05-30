#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from aepsych.config import Config
from aepsych.models.inducing_points.base import BaseAllocator, EMPTY_SIZE
from botorch.utils.sampling import draw_sobol_samples


class SobolAllocator(BaseAllocator):
    """An inducing point allocator that uses Sobol sequences to allocate inducing points."""

    def __init__(self, dim: int, bounds: torch.Tensor) -> None:
        """
        Initializes a Sobol Allocator. This allocator must have bounds.

        Args:
            dim (int): Dimensionality of the search space.
            bounds (torch.Tensor): Bounds for allocating points. Should be of shape
                (2, d).
        """
        # Make sure bounds are the right type so the outputs are the right type
        self.bounds = bounds.to(torch.float64)
        super().__init__(
            dim=dim,
        )

    def allocate_inducing_points(
        self,
        inputs: torch.Tensor | None = None,
        covar_module: torch.nn.Module | None = None,
        num_inducing: int = 100,
        input_batch_shape: torch.Size = EMPTY_SIZE,
    ) -> torch.Tensor:
        """
        Generates `num_inducing` inducing points within the specified bounds using Sobol sampling.

        Args:
            inputs (torch.Tensor): Input tensor, ignored for Sobol points.
            covar_module (torch.nn.Module, optional): Kernel covariance module; included for API compatibility, but not used here.
            num_inducing (int, optional): The number of inducing points to generate. Defaults to 100.
            input_batch_shape (torch.Size, optional): Batch shape, defaults to an empty size; included for API compatibility, but not used here.


        Returns:
            torch.Tensor: A (num_inducing, d)-dimensional tensor of inducing points within the specified bounds.
        """
        # TODO: Usually, these are initialized such that the transforms are applied to
        # bounds, this means that if the transforms change over training, the inducing
        # points aren't in the space. However, we don't have any changing transforms
        # right now.

        # Generate Sobol samples within the unit cube [0,1]^d and rescale to [bounds[0], bounds[1]]
        inducing_points = draw_sobol_samples(
            bounds=self.bounds, n=num_inducing, q=1
        ).squeeze()

        # Ensure correct shape in case Sobol sampling returns a 1D tensor
        if inducing_points.ndim == 1:
            inducing_points = inducing_points.view(-1, 1)

        self.last_allocator_used = self.__class__

        return inducing_points

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get configuration options for the FixedAllocator.

        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the allocator, defaults to None.
            options (dict[str, Any], optional): Additional options, defaults to None.

        Returns:
            dict[str, Any]: Configuration options for the FixedAllocator.
        """
        options = super().get_config_options(config=config, name=name, options=options)

        if "bounds" not in options:
            lb = config.gettensor("common", "lb")
            ub = config.gettensor("common", "ub")
            options["bounds"] = torch.stack((lb, ub))

        return options
