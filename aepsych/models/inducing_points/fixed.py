from typing import Any, Dict, Optional

import torch
from aepsych.config import Config
from aepsych.models.inducing_points.base import BaseAllocator, DummyAllocator


class FixedAllocator(BaseAllocator):
    def __init__(
        self, points: torch.Tensor, bounds: Optional[torch.Tensor] = None
    ) -> None:
        """Initialize the FixedAllocator with inducing points and bounds.

        Args:
            points (torch.Tensor): Inducing points to use.
            bounds (torch.Tensor, optional): Bounds for allocating points. Should be of shape (2, d).
        """
        super().__init__(bounds=bounds)
        self.points = points

    def _get_quality_function(self) -> None:
        """FixedAllocator does not require a quality function, so this returns None."""
        return None

    def allocate_inducing_points(
        self,
        inputs: Optional[torch.Tensor] = None,
        covar_module: Optional[torch.nn.Module] = None,
        num_inducing: int = 10,
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

        self.allocator_used = self.__class__.__name__
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
        if name is None:
            name = cls.__name__
        lb = config.gettensor("common", "lb")
        ub = config.gettensor("common", "ub")
        bounds = torch.stack((lb, ub))
        num_inducing = config.getint("common", "num_inducing", fallback=99)
        fallback_allocator = config.getobj(
            name, "fallback_allocator", fallback=DummyAllocator(bounds=bounds)
        )
        points = config.gettensor(
            name,
            "points",
            fallback=fallback_allocator.allocate_inducing_points(
                num_inducing=num_inducing
            ),
        )
        return {"points": points, "bounds": bounds}
