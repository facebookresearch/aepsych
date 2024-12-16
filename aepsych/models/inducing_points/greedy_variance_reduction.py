from typing import Optional

import torch
from aepsych.models.inducing_points.base import BaseAllocator
from botorch.models.utils.inducing_point_allocators import (
    GreedyVarianceReduction as BaseGreedyVarianceReduction,
)


class GreedyVarianceReduction(BaseGreedyVarianceReduction, BaseAllocator):
    def allocate_inducing_points(
        self,
        inputs: Optional[torch.Tensor] = None,
        covar_module: Optional[torch.nn.Module] = None,
        num_inducing: int = 100,
        input_batch_shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """Allocate inducing points using the GreedyVarianceReduction strategy. This is
        a thin wrapper around BoTorch's GreedyVarianceRedution inducing point allocator.

        Args:
            inputs (torch.Tensor): Input tensor, not required for GreedyVarianceReduction.
            covar_module (torch.nn.Module, optional): Kernel covariance module; included for API compatibility, but not used here.
            num_inducing (int, optional): The number of inducing points to generate. Defaults to 10.
            input_batch_shape (torch.Size, optional): Batch shape, defaults to an empty size; included for API compatibility, but not used here.

        Returns:
            torch.Tensor: The allocated inducing points.
        """
        if inputs is None:  # Dummy points
            return self._allocate_dummy_points(num_inducing=num_inducing)
        else:
            self.last_allocator_used = self.__class__
            return BaseGreedyVarianceReduction.allocate_inducing_points(
                self,
                inputs=inputs,
                covar_module=covar_module,
                num_inducing=num_inducing,
                input_batch_shape=input_batch_shape,
            )
