from typing import Any, Dict, Optional

import torch
from aepsych.config import Config, ConfigurableMixin
from aepsych.models.inducing_points.base import DummyAllocator
from botorch.models.utils.inducing_point_allocators import (
    GreedyVarianceReduction as BaseGreedyVarianceReduction,
)


class GreedyVarianceReduction(BaseGreedyVarianceReduction, ConfigurableMixin):
    def __init__(self, bounds: Optional[torch.Tensor] = None) -> None:
        """Initialize the GreedyVarianceReduction with bounds.

        Args:
            bounds (torch.Tensor, optional): Bounds for allocating points. Should be of shape (2, d).
        """
        super().__init__()

        self.bounds = bounds
        if bounds is not None:
            self.dummy_allocator = DummyAllocator(bounds)
        self.dim = self._initialize_dim()

    def _initialize_dim(self) -> Optional[int]:
        """Initialize the dimension `dim` based on the bounds, if available.

        Returns:
            int: The dimension `d` if bounds are provided, or None otherwise.
        """
        if self.bounds is not None:
            assert self.bounds.shape[0] == 2, "Bounds must have shape (2, d)!"
            lb, ub = self.bounds[0], self.bounds[1]
            for i, (l, u) in enumerate(zip(lb, ub)):
                assert (
                    l <= u
                ), f"Lower bound {l} is not less than or equal to upper bound {u} on dimension {i}!"
            return self.bounds.shape[1]
        return None

    def allocate_inducing_points(
        self,
        inputs: Optional[torch.Tensor] = None,
        covar_module: Optional[torch.nn.Module] = None,
        num_inducing: int = 10,
        input_batch_shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """Allocate inducing points using the GreedyVarianceReduction strategy.

        Args:
            inputs (torch.Tensor): Input tensor, not required for GreedyVarianceReduction.
            covar_module (torch.nn.Module, optional): Kernel covariance module; included for API compatibility, but not used here.
            num_inducing (int, optional): The number of inducing points to generate. Defaults to 10.
            input_batch_shape (torch.Size, optional): Batch shape, defaults to an empty size; included for API compatibility, but not used here.

        Returns:
            torch.Tensor: The allocated inducing points.
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
        else:
            self.allocator_used = self.__class__.__name__
            return super().allocate_inducing_points(
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
        """Get configuration options for the GreedyVarianceReduction allocator.

        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the allocator, defaults to None.
            options (Dict[str, Any], optional): Additional options, defaults to None.

        Returns:
            Dict[str, Any]: Configuration options for the GreedyVarianceReduction allocator.
        """
        if name is None:
            name = cls.__name__
        lb = config.gettensor("common", "lb")
        ub = config.gettensor("common", "ub")
        bounds = torch.stack((lb, ub))
        return {"bounds": bounds}
