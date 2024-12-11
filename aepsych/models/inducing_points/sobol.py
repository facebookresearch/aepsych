from typing import Optional

import torch
from aepsych.models.inducing_points.base import BaseAllocator
from botorch.utils.sampling import draw_sobol_samples


class SobolAllocator(BaseAllocator):
    """An inducing point allocator that uses Sobol sequences to allocate inducing points."""

    dim: int

    def __init__(self, bounds: torch.Tensor, *args, **kwargs) -> None:
        """
        Initializes a Sobol Allocator. This allocator must have bounds.

        Args:
            bounds (torch.Tensor): Bounds for allocating points. Should be of shape
                (2, d).
            *args, **kwargs: Ignores other arguments as only bound are needed for Sobol
                sampling.
        """
        if bounds is None:
            raise ValueError("SobolAllocator must be initialized with bounds.")
        elif bounds.shape[0] != 2:
            raise ValueError(
                "Bounds should be (2, d) in shape to represent lower and upper bounds"
            )
        else:
            # Make sure bounds are the right type so the outputs are the right type
            bounds = bounds.to(torch.float64)
            super().__init__(bounds=bounds)

    def allocate_inducing_points(
        self,
        inputs: Optional[torch.Tensor] = None,
        covar_module: Optional[torch.nn.Module] = None,
        num_inducing: int = 100,
        input_batch_shape: torch.Size = torch.Size([]),
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
        # Generate Sobol samples within the unit cube [0,1]^d and rescale to [bounds[0], bounds[1]]
        inducing_points = draw_sobol_samples(
            bounds=self.bounds, n=num_inducing, q=1
        ).squeeze()

        # Ensure correct shape in case Sobol sampling returns a 1D tensor
        if inducing_points.ndim == 1:
            inducing_points = inducing_points.view(-1, 1)

        self.last_allocator_used = self.__class__

        return inducing_points
