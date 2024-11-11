#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from botorch.utils.sampling import draw_sobol_samples
from scipy.cluster.vq import kmeans2
import torch
from typing import Any, Dict, Optional, Union
import numpy as np

from aepsych.config import Config, ConfigurableMixin

class SobolAllocator(InducingPointAllocator, ConfigurableMixin):
    """An inducing point allocator that uses Sobol sequences to allocate inducing points."""

    def __init__(self, bounds) -> None:
        """Initialize the SobolAllocator without bounds."""
        self.bounds = bounds
        super().__init__()

    

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
        assert self.bounds.shape[0] == 2, "Bounds must have shape (2, d) for Sobol sampling."

        # Generate Sobol samples within the unit cube [0,1]^d and rescale to [bounds[0], bounds[1]]
        inducing_points = draw_sobol_samples(bounds=self.bounds, n=num_inducing, q=1).squeeze()

        # Ensure correct shape in case Sobol sampling returns a 1D tensor
        if inducing_points.ndim == 1:
            inducing_points = inducing_points.view(-1, 1)

        return inducing_points   
    
    @classmethod
    def from_config(cls, config: Config, name: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> "SobolAllocator":
        """Initialize a SobolAllocator from a configuration object.
        
        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the allocator, defaults to None.
            options (Dict[str, Any], optional): Additional options, defaults to None.
            
        Returns:
            SobolAllocator: A SobolAllocator instance.
        """
        return cls(**cls.get_config_options(config, name, options))

    @classmethod
    def get_config_options(cls, config: Config, name: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        bounds = config.gettensor(name, "bounds")
        return {"bounds": bounds}

class KMeansAllocator(InducingPointAllocator, ConfigurableMixin):

    """An inducing point allocator that uses k-means++ to allocate inducing points."""

    def __init__(self) -> None:
        """Initialize the KMeansAllocator."""
        super().__init__()

    def _get_quality_function(self) -> None:
        """K-means++ does not require a quality function, so this returns None."""
        return None
    


    def allocate_inducing_points(
        self,
        inputs: torch.Tensor,
        covar_module: Optional[torch.nn.Module] = None,
        num_inducing: int = 10,
        input_batch_shape: torch.Size = torch.Size([])
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
        # Ensure inputs are unique to avoid duplication issues with k-means++
        unique_inputs = torch.unique(inputs, dim=0)
        
        # If unique inputs are less than or equal to the required inducing points, return them directly
        if unique_inputs.shape[0] <= num_inducing:
            return unique_inputs

        # Run k-means++ on the unique inputs to select inducing points
        inducing_points = torch.tensor(
            kmeans2(unique_inputs.numpy(), num_inducing, minit="++")[0],
            dtype=inputs.dtype,
        )

        return inducing_points
    @classmethod
    def from_config(cls, config: Config, name: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> "KMeansAllocator":
        """Initialize a KMeansAllocator from a configuration object.
        
        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the allocator, defaults to None.
            options (Dict[str, Any], optional): Additional options, defaults to None.
            
        Returns:
            KMeansAllocator: A KMeansAllocator instance."""
        return cls(**cls.get_config_options(config, name, options))

    @classmethod
    def get_config_options(cls, config: Config, name: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        return {}

class AutoAllocator(InducingPointAllocator, ConfigurableMixin):
    """An inducing point allocator that dynamically chooses an allocation strategy
    based on the number of unique data points available."""

    def __init__(self, fallback_allocator: InducingPointAllocator = KMeansAllocator()) -> None:
        """
        Initialize the AutoAllocator with a fallback allocator.

        Args:
            fallback_allocator (InducingPointAllocator, optional): Allocator to use if there are 
                                                        more unique points than required.
        """
        super().__init__()
        self.fallback_allocator = fallback_allocator
        

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
        if inputs is None:
            raise ValueError("Input data must be provided to allocate inducing points.")
        unique_inputs = torch.unique(inputs, dim=0)
        
        # If there are fewer unique points than required, return unique inputs directly
        if unique_inputs.shape[0] <= num_inducing:
            return unique_inputs

        # Otherwise, fall back to the provided allocator (e.g., KMeansAllocator)
        
        return self.fallback_allocator.allocate_inducing_points(
            inputs=inputs,
            covar_module=covar_module,
            num_inducing=num_inducing,
            input_batch_shape=input_batch_shape,
        )
    @classmethod
    def from_config(cls, config: Config, name: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> "AutoAllocator":
        """Initialize an AutoAllocator from a configuration object.
        
        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the allocator, defaults to None.
            options (Dict[str, Any], optional): Additional options, defaults to None.
            
        Returns:
            AutoAllocator: An AutoAllocator instance.
        """
        return cls(**cls.get_config_options(config, name, options))

    @classmethod
    def get_config_options(cls, config: Config, name: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        fallback_allocator_cls = config.getobj(name, "fallback_allocator", fallback=KMeansAllocator)
        fallback_allocator = fallback_allocator_cls.from_config(config) if hasattr(fallback_allocator_cls, 'from_config') else fallback_allocator_cls()
        return {"fallback_allocator": fallback_allocator}

