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
from torch import Tensor

from .rejection_sampler import RejectionSampler

class MonotonicMCAcquisition(AcquisitionFunction):
    """
    Acquisition function for use with the rejection sampling monotonic GP.
    """

    def __init__(
        self,
        model: Model,  # MixedDerivativeVariationalGP
        deriv_constraint_points: torch.Tensor,
        num_samples: int = 32,
        num_rejection_samples: int = 1024,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
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

    def forward(self, X: Tensor) -> Tensor:
        # X is (b) x q x d.
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
        obj_samples = self.objective(samples, X=X)
        return self.acquisition(obj_samples)

    def _set_sampler(self, Xshape: torch.Size) -> None:
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
        model: Model,  # MixedDerivativeVariationalGP
        deriv_constraint_points: torch.Tensor,
        target: float,
        num_samples: int = 32,
        num_rejection_samples: int = 1024,
        beta: float = 3.84,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
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
        mean = obj_samples.mean(dim=0)
        variance = obj_samples.var(dim=0)
        # prevent numerical issues if probit makes all the values 1 or 0
        variance = torch.clamp(variance, min=1e-5)
        delta = torch.sqrt(self.beta * variance)
        return delta - torch.abs(mean - self.target)
