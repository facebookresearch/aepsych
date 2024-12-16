from typing import Any, Dict, Optional

import torch
from aepsych.config import Config
from aepsych.models.inducing_points.base import BaseAllocator
from botorch.utils.sampling import draw_sobol_samples


class SobolAllocator(BaseAllocator):
    """An inducing point allocator that uses Sobol sequences to allocate inducing points."""

    def __init__(self, bounds: torch.Tensor) -> None:
        """Initialize the SobolAllocator with bounds."""
        self.bounds: torch.Tensor = bounds
        super().__init__(bounds=bounds)

    def _get_quality_function(self) -> None:
        """Sobol sampling does not require a quality function, so this returns None."""
        return None

    def allocate_inducing_points(
        self,
        inputs: Optional[torch.Tensor] = None,
        covar_module: Optional[torch.nn.Module] = None,
        num_inducing: int = 10,
        input_batch_shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """
        Generates `num_inducing` inducing points within the specified bounds using Sobol sampling.

        Args:
            inputs (torch.Tensor): Input tensor, not required for Sobol sampling.
            covar_module (torch.nn.Module, optional): Kernel covariance module; included for API compatibility, but not used here.
            num_inducing (int, optional): The number of inducing points to generate. Defaults to 10.
            input_batch_shape (torch.Size, optional): Batch shape, defaults to an empty size; included for API compatibility, but not used here.


        Returns:
            torch.Tensor: A (num_inducing, d)-dimensional tensor of inducing points within the specified bounds.

        Raises:
            ValueError: If `bounds` is not provided.
        """
        # TODO: Usually, these are initialized such that the transforms are applied to
        # bounds, this means that if the transforms change over training, the inducing
        # points aren't in the space. However, we don't have any changing transforms
        # right now.

        # Validate bounds shape
        assert (
            self.bounds.shape[0] == 2
        ), "Bounds must have shape (2, d) for Sobol sampling."
        # if bounds are long, make them float
        if self.bounds.dtype == torch.long:
            self.bounds = self.bounds.float()
        # Generate Sobol samples within the unit cube [0,1]^d and rescale to [bounds[0], bounds[1]]
        inducing_points = draw_sobol_samples(
            bounds=self.bounds, n=num_inducing, q=1
        ).squeeze()

        # Ensure correct shape in case Sobol sampling returns a 1D tensor
        if inducing_points.ndim == 1:
            inducing_points = inducing_points.view(-1, 1)
        self.allocator_used = "SobolAllocator"
        return inducing_points

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get configuration options for the SobolAllocator.

        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the allocator, defaults to None.
            options (Dict[str, Any], optional): Additional options, defaults to None.

        Returns:
            Dict[str, Any]: Configuration options for the SobolAllocator.
        """
        if name is None:
            name = cls.__name__
        lb = config.gettensor("common", "lb")
        ub = config.gettensor("common", "ub")
        bounds = torch.stack((lb, ub))
        return {"bounds": bounds}
