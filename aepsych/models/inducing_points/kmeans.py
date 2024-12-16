from typing import Optional

import torch
from aepsych.models.inducing_points.base import BaseAllocator
from scipy.cluster.vq import kmeans2


class KMeansAllocator(BaseAllocator):
    """An inducing point allocator that uses k-means++ to allocate inducing points."""

    def allocate_inducing_points(
        self,
        inputs: Optional[torch.Tensor] = None,
        covar_module: Optional[torch.nn.Module] = None,
        num_inducing: int = 100,
        input_batch_shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """
        Generates `num_inducing` inducing points using k-means++ initialization on the input data.

        Args:
            inputs (torch.Tensor): A tensor of shape (n, d) containing the input data.
            covar_module (torch.nn.Module, optional): Kernel covariance module; included for API compatibility, but not used here.
            num_inducing (int, optional): The number of inducing points to generate. Defaults to 100.
            input_batch_shape (torch.Size, optional): Batch shape, defaults to an empty size; included for API compatibility, but not used here.

        Returns:
            torch.Tensor: A (num_inducing, d)-dimensional tensor of inducing points selected via k-means++.
        """
        if inputs is None:  # Dummy points
            return self._allocate_dummy_points(num_inducing=num_inducing)

        if inputs.shape[1] != self.dim:
            # The inputs were augmented somehow, assuming it was added to the end of dims
            inputs = inputs[:, : self.dim, ...]

        self.last_allocator_used = self.__class__

        # Ensure inputs are unique to avoid duplication issues with k-means++
        unique_inputs = torch.unique(inputs, dim=0)

        # If unique inputs are less than or equal to the required inducing points, return them directly
        if unique_inputs.shape[0] <= num_inducing:
            return unique_inputs

        # Run k-means++ on the unique inputs to select inducing points
        inducing_points = torch.tensor(
            kmeans2(unique_inputs.cpu().numpy(), num_inducing, minit="++")[0]
        )

        return inducing_points
