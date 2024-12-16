from typing import Any, Dict, Optional

import torch
from aepsych.config import Config
from aepsych.models.inducing_points.base import BaseAllocator


class FixedAllocator(BaseAllocator):
    def __init__(
        self,
        dim: int,
        points: torch.Tensor,
    ) -> None:
        """Initialize the FixedAllocator with inducing points to use and bounds.

        Args:
            dim (int): Dimensionality of the search space.
            points (torch.Tensor): Inducing points to use (should be n, d).
        """
        super().__init__(dim=dim)
        self.points = points

    def allocate_inducing_points(
        self,
        inputs: Optional[torch.Tensor] = None,
        covar_module: Optional[torch.nn.Module] = None,
        num_inducing: int = 100,
        input_batch_shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """Allocate inducing points by returning the fixed inducing points.

        Args:
            inputs (torch.Tensor): Input tensor, not required for FixedAllocator.
            covar_module (torch.nn.Module, optional): Kernel covariance module; included for API compatibility, but not used here.
            num_inducing (int, optional): The number of inducing points to generate. Defaults to 10.
            input_batch_shape (torch.Size, optional): Batch shape, defaults to an empty size; included for API compatibility, but not used here.

        Returns:
            torch.Tensor: The fixed inducing points.
        """
        # TODO: Usually, these are initialized such that the transforms are applied to
        # points already, this means that if the transforms change over training, the inducing
        # points aren't in the space. However, we don't have any changing transforms
        # right now.

        self.last_allocator_used = self.__class__
        return self.points

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get configuration options for the FixedAllocator.

        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the allocator, defaults to None.
            options (Dict[str, Any], optional): Additional options, defaults to None.

        Returns:
            Dict[str, Any]: Configuration options for the FixedAllocator.
        """
        options = super().get_config_options(config=config, name=name, options=options)

        options["points"] = config.gettensor("FixedAllocator", "points")

        return options
