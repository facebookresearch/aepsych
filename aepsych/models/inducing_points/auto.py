from typing import Any, Dict, Optional

import torch
from aepsych.config import Config
from aepsych.models.inducing_points.base import BaseAllocator, DummyAllocator
from aepsych.models.inducing_points.kmeans import KMeansAllocator
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator


class AutoAllocator(BaseAllocator):
    """An inducing point allocator that dynamically chooses an allocation strategy
    based on the number of unique data points available."""

    def __init__(
        self,
        bounds: Optional[torch.Tensor] = None,
        fallback_allocator: InducingPointAllocator = KMeansAllocator(),
    ) -> None:
        """
        Initialize the AutoAllocator with a fallback allocator.

        Args:
            fallback_allocator (InducingPointAllocator, optional): Allocator to use if there are
                                                        more unique points than required.
        """
        super().__init__(bounds=bounds)
        self.fallback_allocator = fallback_allocator
        if bounds is not None:
            self.bounds = bounds
            self.dummy_allocator = DummyAllocator(bounds=bounds)

    def _get_quality_function(self) -> None:
        """AutoAllocator does not require a quality function, so this returns None."""
        return None

    def allocate_inducing_points(
        self,
        inputs: Optional[torch.Tensor],
        covar_module: Optional[torch.nn.Module] = None,
        num_inducing: int = 10,
        input_batch_shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """
        Allocate inducing points by either using the unique input data directly
        or falling back to another allocation method if there are too many unique points.

        Args:
            inputs (torch.Tensor): A tensor of shape (n, d) containing the input data.
            covar_module (torch.nn.Module, optional): Kernel covariance module; included for API compatibility, but not used here.
            num_inducing (int, optional): The number of inducing points to generate.
            input_batch_shape (torch.Size, optional): Batch shape, defaults to an empty size; included for API compatibility, but not used here.

        Returns:
            torch.Tensor: A (num_inducing, d)-dimensional tensor of inducing points.
        """
        # Ensure inputs are not None
        if inputs is None and self.bounds is not None:
            self.allocator_used = self.dummy_allocator.__class__.__name__
            return self.dummy_allocator.allocate_inducing_points(
                inputs=inputs,
                covar_module=covar_module,
                num_inducing=num_inducing,
                input_batch_shape=input_batch_shape,
            )
        elif inputs is None and self.bounds is None:
            raise ValueError(f"Either inputs or bounds must be provided.{self.bounds}")

        assert (
            inputs is not None
        ), "inputs should not be None here"  # to make mypy happy

        unique_inputs = torch.unique(inputs, dim=0)

        # If there are fewer unique points than required, return unique inputs directly
        if unique_inputs.shape[0] <= num_inducing:
            self.allocator_used = self.__class__.__name__
            return unique_inputs

        # Otherwise, fall back to the provided allocator (e.g., KMeansAllocator)
        if inputs.shape[0] <= num_inducing:
            self.allocator_used = self.__class__.__name__
            return inputs
        else:
            self.allocator_used = self.fallback_allocator.__class__.__name__
            return self.fallback_allocator.allocate_inducing_points(
                inputs=inputs,
                covar_module=covar_module,
                num_inducing=num_inducing,
                input_batch_shape=input_batch_shape,
            )

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get configuration options for the AutoAllocator.

        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the allocator, defaults to None.
            options (Dict[str, Any], optional): Additional options, defaults to None.

        Returns:
            Dict[str, Any]: Configuration options for the AutoAllocator.
        """
        if name is None:
            name = cls.__name__
        lb = config.gettensor("common", "lb")
        ub = config.gettensor("common", "ub")
        bounds = torch.stack((lb, ub))
        fallback_allocator_cls = config.getobj(
            name, "fallback_allocator", fallback=KMeansAllocator
        )
        fallback_allocator = (
            fallback_allocator_cls.from_config(config)
            if hasattr(fallback_allocator_cls, "from_config")
            else fallback_allocator_cls()
        )

        return {"fallback_allocator": fallback_allocator, "bounds": bounds}
