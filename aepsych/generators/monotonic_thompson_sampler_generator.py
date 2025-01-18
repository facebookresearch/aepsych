#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Dict, List, Optional, Type

import torch
from aepsych.acquisition.objective import ProbitObjective
from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.models.monotonic_rejection_gp import MonotonicRejectionGP
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.utils.sampling import draw_sobol_samples


class MonotonicThompsonSamplerGenerator(AEPsychGenerator[MonotonicRejectionGP]):
    """A generator specifically to be used with MonotonicRejectionGP that uses a Thompson-sampling-style
    approach for gen, rather than using an acquisition function. We draw a posterior sample at a large number
    of points, and then choose the point that is closest to the target value.
    """

    def __init__(
        self,
        n_samples: int,
        n_rejection_samples: int,
        num_ts_points: int,
        target_value: float,
        objective: MCAcquisitionObjective,
        dim: int,
        explore_features: Optional[List[Type[int]]] = None,
    ) -> None:
        """Initialize MonotonicMCAcquisition

        Args:
            n_samples (int): Number of samples to select point from.
            num_rejection_samples (int): Number of rejection samples to draw.
            num_ts_points (int): Number of points at which to sample.
            target_value (float): target value that is being looked for
            objective (MCAcquisitionObjective): Objective transform of the GP output
                before evaluating the acquisition. Defaults to identity transform.
            dim (int): Dimensionality of the model.
            explore_features (List[Type[int]], optional): List of features that will be selected randomly and then
                fixed for acquisition fn optimization. Defaults to None.
        """
        self.n_samples = n_samples
        self.n_rejection_samples = n_rejection_samples
        self.num_ts_points = num_ts_points
        self.target_value = target_value
        self.objective = objective()
        self.explore_features = explore_features
        self.dim = dim

    def gen(
        self,
        num_points: int,  # Current implementation only generates 1 point at a time
        model: MonotonicRejectionGP,
        fixed_features: Optional[Dict[int, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Query next point(s) to run by optimizing the acquisition function.
        Args:
            num_points (int): Number of points to query. current implementation only generates 1 point at a time.
            model (MonotonicRejectionGP): Fitted model of the data.
            fixed_features (Dict[int, float], optional): Not implemented for this generator.
            **kwargs: Ignored, API compatibility
        Returns:
            torch.Tensor: Next set of point(s) to evaluate, [num_points x dim].
        """
        if fixed_features is not None:
            warnings.warn(
                "Cannot fix features when generating from MonotonicRejectionGenerator"
            )
        # Generate the points at which to sample
        X = draw_sobol_samples(bounds=model.bounds_, n=self.num_ts_points, q=1).squeeze(
            1
        )
        # Fix any explore features
        if self.explore_features is not None:
            for idx in self.explore_features:
                val = (
                    model.bounds_[0, idx]  # type: ignore
                    + torch.rand(1) * (model.bounds_[1, idx] - model.bounds_[0, idx])  # type: ignore
                ).item()
                X[:, idx] = val

        # Draw n samples
        f_samp = model.sample(
            X,
            num_samples=self.n_samples,
            num_rejection_samples=self.n_rejection_samples,
        )

        # Find the point closest to target
        dist = torch.abs(self.objective(f_samp) - self.target_value)
        best_indx = torch.argmin(dist, dim=1)
        return torch.Tensor(X[best_indx])

    @classmethod
    def from_config(cls, config: Config) -> "MonotonicThompsonSamplerGenerator":
        """
        Creates an instance of MonotonicThompsonSamplerGenerator from a configuration object.

        Args:
            config (Config): Configuration object containing initialization parameters.

        Returns:
            MonotonicThompsonSamplerGenerator: A configured instance of the generator class with specified number of samples,
            rejection samples, Thompson sampling points, target value, objective transformation, and optional exploration features.
        """
        classname = cls.__name__
        n_samples = config.getint(classname, "num_samples", fallback=1)
        n_rejection_samples = config.getint(
            classname, "num_rejection_samples", fallback=500
        )
        num_ts_points = config.getint(classname, "num_ts_points", fallback=1000)
        target = config.getfloat(classname, "target", fallback=0.75)
        objective = config.getobj(classname, "objective", fallback=ProbitObjective)
        explore_features = config.getlist(
            classname, "explore_idxs", element_type=int, fallback=None
        )  # type: ignore

        return cls(
            n_samples=n_samples,
            n_rejection_samples=n_rejection_samples,
            num_ts_points=num_ts_points,
            target_value=target,
            objective=objective,
            explore_features=explore_features,  # type: ignore
        )
