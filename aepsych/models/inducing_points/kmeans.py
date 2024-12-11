from typing import Any, Dict, Optional

import torch
from aepsych.config import Config
from aepsych.models.inducing_points.base import BaseAllocator, DummyAllocator
from scipy.cluster.vq import kmeans2


class KMeansAllocator(BaseAllocator):
    """An inducing point allocator that uses k-means++ to allocate inducing points."""

    def __init__(self, bounds: Optional[torch.Tensor] = None) -> None:
        """Initialize the KMeansAllocator."""
        super().__init__(bounds=bounds)
        if bounds is not None:
            self.bounds = bounds
            self.dummy_allocator = DummyAllocator(bounds)

    def _get_quality_function(self) -> None:
        """K-means++ does not require a quality function, so this returns None."""
        return None

    def allocate_inducing_points(
        self,
        inputs: Optional[torch.Tensor] = None,
        covar_module: Optional[torch.nn.Module] = None,
        num_inducing: int = 10,
        input_batch_shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """
        Generates `num_inducing` inducing points using k-means++ initialization on the input data.

        Args:
            inputs (torch.Tensor): A tensor of shape (n, d) containing the input data.
            covar_module (torch.nn.Module, optional): Kernel covariance module; included for API compatibility, but not used here.
            num_inducing (int, optional): The number of inducing points to generate. Defaults to 10.
            input_batch_shape (torch.Size, optional): Batch shape, defaults to an empty size; included for API compatibility, but not used here.

        Returns:
            torch.Tensor: A (num_inducing, d)-dimensional tensor of inducing points selected via k-means++.
        """
        if inputs is None and self.bounds is not None:
            self.allocator_used = self.dummy_allocator.__class__.__name__
            return self.dummy_allocator.allocate_inducing_points(
                inputs=inputs,
                covar_module=covar_module,
                num_inducing=num_inducing,
                input_batch_shape=input_batch_shape,
            )
        elif inputs is None and self.bounds is None:
            raise ValueError("Either inputs or bounds must be provided.")
        # Ensure inputs are unique to avoid duplication issues with k-means++
        unique_inputs = torch.unique(inputs, dim=0)

        # If unique inputs are less than or equal to the required inducing points, return them directly
        if unique_inputs.shape[0] <= num_inducing:
            self.allocator_used = self.__class__.__name__
            return unique_inputs

        # Run k-means++ on the unique inputs to select inducing points
        inducing_points = torch.tensor(
            kmeans2(unique_inputs.cpu().numpy(), num_inducing, minit="++")[0]
        )
        self.allocator_used = self.__class__.__name__
        return inducing_points

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get configuration options for the KMeansAllocator.

        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the allocator, defaults to None.
            options (Dict[str, Any], optional): Additional options, defaults to None.

        Returns:
            Dict[str, Any]: Configuration options for the KMeansAllocator.
        """
        if name is None:
            name = cls.__name__
        lb = config.gettensor("common", "lb")
        ub = config.gettensor("common", "ub")
        bounds = torch.stack((lb, ub))
        return {"bounds": bounds}
