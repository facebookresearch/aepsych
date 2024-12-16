from typing import Optional

import torch
from aepsych.models.inducing_points.base import BaseAllocator
from aepsych.models.inducing_points.greedy_variance_reduction import (
    GreedyVarianceReduction,
)


class AutoAllocator(BaseAllocator):
    """An inducing point allocator that theoretically picks the best allocator method
    based on the given data/model."""

    def allocate_inducing_points(
        self,
        inputs: Optional[torch.Tensor] = None,
        covar_module: Optional[torch.nn.Module] = None,
        num_inducing: int = 100,
        input_batch_shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """Generates `num_inducing` inducing points smartly based on the inputs.
        Currently, this is just a wrapper for the greedy variance allocator

        Args:
            inputs (torch.Tensor): A tensor of shape (n, d) containing the input data.
            covar_module (torch.nn.Module, optional): Kernel covariance module; included for API compatibility, but not used here.
            num_inducing (int, optional): The number of inducing points to generate. Defaults to 100.
            input_batch_shape (torch.Size, optional): Batch shape, defaults to an empty size; included for API compatibility, but not used here.

        Returns:
            torch.Tensor: A (num_inducing, d)-dimensional tensor of inducing points selected via k-means++.
        """
        # Auto allocator actually just wraps the greedy variance allocator
        allocator = GreedyVarianceReduction(dim=self.dim)

        points = allocator.allocate_inducing_points(
            inputs=inputs,
            covar_module=covar_module,
            num_inducing=num_inducing,
            input_batch_shape=input_batch_shape,
        )

        self.last_allocator_used = allocator.last_allocator_used
        return points
