from typing import Any, Dict, Optional

import torch
from aepsych.config import Config
from aepsych.models.inducing_points.base import BaseAllocator


class FixedAllocator(BaseAllocator):
    def __init__(
        self,
        bounds: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the FixedAllocator with inducing points to use and bounds.

        Args:
            bounds (torch.Tensor, optional): Bounds for allocating points. Should be of
                shape (2, d). Here for API uniformity, ignored for points.
            points (torch.Tensor, optional): Inducing points to use (should be n, d).
                Not actually optional, must be set, but Optional for API uniformity.
            *args, **kwargs: Ignores other arguments.
        """
        if points is None:
            raise ValueError("points must be set to initialize the fixed allocator.")
        else:
            dim = points.shape[1]

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
