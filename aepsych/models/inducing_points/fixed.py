from typing import Any

import torch
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
        inputs: torch.Tensor | None = None,
        covar_module: torch.nn.Module | None = None,
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


class FixedPlusAllocator(BaseAllocator):
    def __init__(
        self,
        dim: int,
        points: torch.Tensor,
        main_allocator: type[BaseAllocator] | BaseAllocator,
        **kwargs: Any,
    ) -> None:
        """Initialize the FixedPlusAllocator where inducing points are first
        created by another inducing point algorithm, then the fixed points are
        added if they are not already in the set.

        Args:
            dim (int): Dimensionality of the search space.
            points (torch.Tensor): Inducing points to use (should be n, d).
            main_allocator (type[BaseAllocator] | BaseAllocator): The inducing point
                algorithm to use first. If it is a type, initialize it with the args
                passed in, otherwise, just wrap it.
            **kwargs: Keyword arguments to pass to the main_allocator to
                initialize it.
        """
        if points.shape[1] != dim:
            raise ValueError(
                "Points must have the same dimensionality as the search space. "
                f"Points have {points.shape[1]} dimensions, but the search "
                f"space has {dim} dimensions."
            )

        super().__init__(dim=dim)
        self.extra_points = points
        if isinstance(main_allocator, type):
            self.main_allocator = main_allocator(dim=dim, **kwargs)
        else:
            self.main_allocator = main_allocator

    def allocate_inducing_points(
        self,
        inputs: torch.Tensor | None = None,
        covar_module: torch.nn.Module | None = None,
        num_inducing: int = 100,
        input_batch_shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        points = self.main_allocator.allocate_inducing_points(
            inputs=inputs,
            covar_module=covar_module,
            num_inducing=num_inducing,
            input_batch_shape=input_batch_shape,
        )

        # Check if points are dummy, if so don't bother
        if self.main_allocator.last_allocator_used is None:
            return points

        # Concatenate the fixed points to the main points
        points = torch.cat([points, self.extra_points], dim=0)

        # Apply unique to remove duplicates
        points = torch.unique(points, dim=0)

        self.last_allocator_used = self.__class__
        return points
