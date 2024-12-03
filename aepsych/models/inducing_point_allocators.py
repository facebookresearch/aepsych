#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from aepsych.config import Config, ConfigurableMixin
from aepsych.utils import get_bounds
from botorch.models.utils.inducing_point_allocators import (
    GreedyVarianceReduction as BaseGreedyVarianceReduction,
    InducingPointAllocator,
)
from botorch.utils.sampling import draw_sobol_samples
from scipy.cluster.vq import kmeans2


class BaseAllocator(InducingPointAllocator, ConfigurableMixin):
    """Base class for inducing point allocators."""

    def __init__(self, bounds: Optional[torch.Tensor] = None) -> None:
        """
        Initialize the allocator with optional bounds.

        Args:
            bounds (torch.Tensor, optional): Bounds for allocating points. Should be of shape (2, d).
        """
        self.bounds = bounds
        self.dim = self._initialize_dim()

    def _initialize_dim(self) -> Optional[int]:
        """
        Initialize the dimension `dim` based on the bounds, if available.

        Returns:
            int: The dimension `d` if bounds are provided, or None otherwise.
        """
        if self.bounds is not None:
            # Validate bounds and extract dimension
            assert self.bounds.shape[0] == 2, "Bounds must have shape (2, d)!"
            lb, ub = self.bounds[0], self.bounds[1]
            for i, (l, u) in enumerate(zip(lb, ub)):
                assert (
                    l <= u
                ), f"Lower bound {l} is not less than or equal to upper bound {u} on dimension {i}!"
            return self.bounds.shape[1]  # Number of dimensions (d)
        return None

    def _determine_dim_from_inputs(self, inputs: torch.Tensor) -> int:
        """
        Determine dimension `dim` from the inputs tensor.

        Args:
            inputs (torch.Tensor): Input tensor of shape (..., d).

        Returns:
            int: The inferred dimension `d`.
        """
        return inputs.shape[-1]

    @abstractmethod
    def allocate_inducing_points(
        self,
        inputs: Optional[torch.Tensor],
        covar_module: Optional[torch.nn.Module],
        num_inducing: int,
        input_batch_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Abstract method for allocating inducing points.

        Args:
            inputs (torch.Tensor, optional): Input tensor, implementation-specific.
            covar_module (torch.nn.Module, optional): Kernel covariance module.
            num_inducing (int): Number of inducing points to allocate.
            input_batch_shape (torch.Size): Shape of the input batch.

        Returns:
            torch.Tensor: Allocated inducing points.
        """
        if self.dim is None and inputs is not None:
            self.dim = self._determine_dim_from_inputs(inputs)

        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def _get_quality_function(self) -> Optional[Any]:
        """
        Abstract method for returning a quality function if required.

        Returns:
            None or Callable: Quality function if needed.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


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
        bounds = get_bounds(config)
        return {"bounds": bounds}


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
            return unique_inputs

        # Run k-means++ on the unique inputs to select inducing points
        inducing_points = torch.tensor(
            kmeans2(unique_inputs.cpu().numpy(), num_inducing, minit="++")[0]
        )

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
        bounds = get_bounds(config)
        return {"bounds": bounds}


class DummyAllocator(BaseAllocator):
    def __init__(self, bounds: torch.Tensor) -> None:
        """Initialize the DummyAllocator with bounds.

        Args:
            bounds (torch.Tensor): Bounds for allocating points. Should be of shape (2, d).
        """
        super().__init__(bounds=bounds)
        self.bounds: torch.Tensor = bounds

    def _get_quality_function(self) -> None:
        """DummyAllocator does not require a quality function, so this returns None."""
        return None

    def allocate_inducing_points(
        self,
        inputs: Optional[torch.Tensor] = None,
        covar_module: Optional[torch.nn.Module] = None,
        num_inducing: int = 10,
        input_batch_shape: torch.Size = torch.Size([]),
    ) -> torch.Tensor:
        """Allocate inducing points by returning zeros of the appropriate shape.

        Args:
            inputs (torch.Tensor): Input tensor, not required for DummyAllocator.
            covar_module (torch.nn.Module, optional): Kernel covariance module; included for API compatibility, but not used here.
            num_inducing (int, optional): The number of inducing points to generate. Defaults to 10.
            input_batch_shape (torch.Size, optional): Batch shape, defaults to an empty size; included for API compatibility, but not used here.

        Returns:
            torch.Tensor: A (num_inducing, d)-dimensional tensor of zeros.
        """
        return torch.zeros(num_inducing, self.bounds.shape[-1])

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get configuration options for the DummyAllocator.

        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the allocator, defaults to None.
            options (Dict[str, Any], optional): Additional options, defaults to None.

        Returns:
            Dict[str, Any]: Configuration options for the DummyAllocator.
        """
        if name is None:
            name = cls.__name__
        bounds = get_bounds(config)
        return {"bounds": bounds}


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
            return unique_inputs

        # Otherwise, fall back to the provided allocator (e.g., KMeansAllocator)
        if inputs.shape[0] <= num_inducing:
            return inputs
        else:
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
        bounds = get_bounds(config)
        fallback_allocator_cls = config.getobj(
            name, "fallback_allocator", fallback=KMeansAllocator
        )
        fallback_allocator = (
            fallback_allocator_cls.from_config(config)
            if hasattr(fallback_allocator_cls, "from_config")
            else fallback_allocator_cls()
        )

        return {"fallback_allocator": fallback_allocator, "bounds": bounds}


class FixedAllocator(BaseAllocator):
    def __init__(
        self, inducing_points: torch.Tensor, bounds: Optional[torch.Tensor] = None
    ) -> None:
        """Initialize the FixedAllocator with inducing points and bounds.

        Args:
            inducing_points (torch.Tensor): Inducing points to use.
            bounds (torch.Tensor, optional): Bounds for allocating points. Should be of shape (2, d).
        """
        super().__init__(bounds=bounds)
        self.inducing_points = inducing_points
        self.bounds = bounds

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
        return self.inducing_points

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
        bounds = torch.stack([lb, ub])
        num_inducing = config.getint("common", "num_inducing")
        fallback_allocator = config.getobj(
            name, "fallback_allocator", fallback=DummyAllocator(bounds=bounds)
        )
        inducing_points = config.gettensor(
            name,
            "inducing_points",
            fallback=fallback_allocator.allocate_inducing_points(
                num_inducing=num_inducing
            ),
        )
        return {"inducing_points": inducing_points}


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
            return self.dummy_allocator.allocate_inducing_points(
                inputs=inputs,
                covar_module=covar_module,
                num_inducing=num_inducing,
                input_batch_shape=input_batch_shape,
            )
        elif inputs is None and self.bounds is None:
            raise ValueError("Either inputs or bounds must be provided.")
        else:
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
        bounds = get_bounds(config)
        return {"bounds": bounds}
