from abc import abstractmethod
from typing import Any

import torch
from aepsych.config import Config, ConfigurableMixin
from aepsych.utils import get_dims
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator


class BaseAllocator(InducingPointAllocator, ConfigurableMixin):
    """Base class for inducing point allocators."""

    def __init__(self, dim: int, *args, **kwargs) -> None:
        """
        Initialize the allocator with optional bounds.

        Args:
            dim (int): Dimensionality of the search space.
            *args, **kwargs: Other allocator specific arguments.
        """
        self.dim = dim
        self.last_allocator_used: InducingPointAllocator | None = None

    @abstractmethod
    def allocate_inducing_points(
        self,
        inputs: torch.Tensor | None = None,
        covar_module: torch.nn.Module | None = None,
        num_inducing: int = 100,
        input_batch_shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """
        Abstract method for allocating inducing points. Must replace the
        last_allocator_used attribute for what was actually used to produce the
        inducing points. Dummy points should be made when it is not possible to create
        inducing points (e.g., inputs is None).

        Args:
            inputs (torch.Tensor, optional): Input tensor, implementation-specific.
            covar_module (torch.nn.Module, optional): Kernel covariance module.
            num_inducing (int): Number of inducing points to allocate.
            input_batch_shape (torch.Size): Shape of the input batch.

        Returns:
            torch.Tensor: Allocated inducing points.
        """

    def _allocate_dummy_points(self, num_inducing: int = 100) -> torch.Tensor:
        """Return dummy inducing points with the correct dimensionality.

        Args:
            num_inducing (int): Number of inducing points to make, defaults to 100.
        """
        self.last_allocator_used = None
        return torch.zeros(num_inducing, self.dim)

    def _get_quality_function(self):
        return super()._get_quality_function()

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get configuration options for the allocator.

        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the allocator, defaults to None. Ignored.
            options (dict[str, Any], optional): Additional options, defaults to None.

        Returns:
            dict[str, Any]: Configuration options for the DummyAllocator.
        """
        options = super().get_config_options(config, name, options)

        if "dim" not in options:
            options["dim"] = get_dims(config)

        return options
