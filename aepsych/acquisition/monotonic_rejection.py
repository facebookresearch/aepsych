#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective, MCAcquisitionObjective
from botorch.models.model import Model

from .rejection_sampler import RejectionSampler


class MonotonicMCAcquisition(AcquisitionFunction):
    """
    Acquisition function base class for use with the rejection sampling
        monotonic GP. This handles the bookkeeping of the derivative
        constraint points -- implement specific monotonic MC acquisition
        in subclasses.
    """

    def __init__(
        self,
        model: Model,
        deriv_constraint_points: torch.Tensor,
        num_samples: int = 32,
        num_rejection_samples: int = 1024,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        """Initialize MonotonicMCAcquisition

        Args:
            model (Model): Model to use, usually a MonotonicRejectionGP.
            num_samples (int): Number of samples to keep from the rejection sampler. Defaults to 32.
            num_rejection_samples (int): Number of rejection samples to draw. Defaults to 1024.
            objective (MCAcquisitionObjective, optional): Objective transform of the GP output
                before evaluating the acquisition. Defaults to identity transform.
        """
        super().__init__(model=model)
        self.deriv_constraint_points = deriv_constraint_points
        self.num_samples = num_samples
        self.num_rejection_samples = num_rejection_samples
        self.sampler_shape = torch.Size([])
        if objective is None:
            assert model.num_outputs == 1
            objective = IdentityMCObjective()
        else:
            assert isinstance(objective, MCAcquisitionObjective)
        self.add_module("objective", objective)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the acquisition function at a set of points.

        Args:
            X (torch.Tensor): Points at which to evaluate the acquisition function.
                Should be (b) x q x d, and q should be 1.

        Returns:
            torch.Tensor: Acquisition function value at these points.
        """
        # This is currently doing joint samples over (b), and requiring q=1
        # TODO T68656582 support batches properly.
        if len(X.shape) == 3:
            assert X.shape[1] == 1, "q must be 1"
            Xfull = torch.cat((X[:, 0, :], self.deriv_constraint_points), dim=0)
        else:
            Xfull = torch.cat((X, self.deriv_constraint_points), dim=0)
        if not hasattr(self, "sampler") or Xfull.shape != self.sampler_shape:
            self._set_sampler(X.shape)
            self.sampler_shape = Xfull.shape
        posterior = self.model.posterior(Xfull)
        samples = self.sampler(posterior)
        assert len(samples.shape) == 3
        # Drop derivative samples
        samples = samples[:, : X.shape[0], :]
        # NOTE: Squeeze below makes sure that we pass in the same `X` that was used
        # to generate the `samples`. This is necessitated by `MCAcquisitionObjective`,
        # which verifies that `samples` and `X` have the same q-batch size.
        obj_samples = self.objective(samples, X=X.squeeze(-2) if X.ndim == 3 else X)
        return self.acquisition(obj_samples)

    def _set_sampler(self, Xshape: torch.Size) -> None:
        """
        Sets up the rejection sampler for generating samples with derivative constraints.

        Args:
            Xshape (torch.Size): The shape of the input points `X` for which the sampler is set up.

        """
        sampler = RejectionSampler(
            num_samples=self.num_samples,
            num_rejection_samples=self.num_rejection_samples,
            constrained_idx=torch.arange(
                Xshape[0], Xshape[0] + self.deriv_constraint_points.shape[0]
            ),
        )
        self.add_module("sampler", sampler)

    def acquisition(self, obj_samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MonotonicMCLSE(MonotonicMCAcquisition):
    def __init__(
        self,
        model: Model,
        deriv_constraint_points: torch.Tensor,
        target: float,
        num_samples: int = 32,
        num_rejection_samples: int = 1024,
        beta: float = 3.84,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        """Level set estimation acquisition function for use with monotonic models.

        Args:
            model (Model): Underlying model object, usually should be MonotonicRejectionGP.
            deriv_constraint_points (torch.Tensor): Points at which the derivative should be constrained.
            target (float): Level set value to target (after the objective).
            num_samples (int): Number of MC samples to draw in MC acquisition. Defaults to 32.
            num_rejection_samples (int): Number of rejection samples from which to subsample monotonic ones. Defaults to 1024.
            beta (float): Parameter of the LSE acquisition function that governs exploration vs
                exploitation (similarly to the same parameter in UCB). Defaults to 3.84 (1.96 ** 2), which maps to the straddle
                heuristic of Bryan et al. 2005.
            objective (MCAcquisitionObjective, optional): Objective transform. Defaults to identity transform.
        """
        self.beta = beta
        self.target = target
        super().__init__(
            model=model,
            deriv_constraint_points=deriv_constraint_points,
            num_samples=num_samples,
            num_rejection_samples=num_rejection_samples,
            objective=objective,
        )

    def acquisition(self, obj_samples: torch.Tensor) -> torch.Tensor:
        """
        Computes the acquisition function value for level set estimation in monotonic models.

        Args:
            obj_samples (torch.Tensor): Tensor of samples from the model, transformed by the objective.
                Expected shape is samples x batch_shape.

        Returns:
            torch.Tensor: The acquisition function value, calculated as the difference between an exploration-exploitation term
            (based on the variance and `beta` parameter) and the absolute difference between the mean and the target level set.
        """
        mean = obj_samples.mean(dim=0)
        variance = obj_samples.var(dim=0)
        # prevent numerical issues if probit makes all the values 1 or 0
        variance = torch.clamp(variance, min=1e-5)
        delta = torch.sqrt(self.beta * variance)
        return delta - torch.abs(mean - self.target)
