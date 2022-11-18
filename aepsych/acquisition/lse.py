#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
from aepsych.acquisition.objective import ProbitObjective
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
    MCAcquisitionObjective,
    MCSampler,
)
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class MCLevelSetEstimation(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        target: Union[float, Tensor] = 0.75,
        beta: Union[float, Tensor] = 3.84,
        objective: Optional[MCAcquisitionObjective] = None,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        r"""Monte-carlo level set estimation.

        Args:
            model: A fitted model.
            target: the level set (after objective transform) to be estimated
            beta: a parameter that governs explore-exploit tradeoff
            objective: An MCAcquisitionObjective representing the link function
                (e.g., logistic or probit.) applied on the samples.
                Can be implemented via GenericMCObjective.
            sampler: The sampler used for drawing MC samples.
        """
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        if objective is None:
            objective = ProbitObjective()
        super().__init__(model=model, sampler=sampler, objective=None, X_pending=None)
        self.objective = objective
        self.beta = beta
        self.target = target

    def acquisition(self, obj_samples: torch.Tensor) -> torch.Tensor:
        """Evaluate the acquisition based on objective samples.

        Usually you should not call this directly unless you are
        subclassing this class and modifying how objective samples
        are generated.

        Args:
            obj_samples (torch.Tensor): Samples from the model, transformed
                by the objective. Should be samples x batch_shape.

        Returns:
            torch.Tensor: Acquisition function at the sampled values.
        """
        mean = obj_samples.mean(dim=0)
        variance = obj_samples.var(dim=0)
        # prevent numerical issues if probit makes all the values 1 or 0
        variance = torch.clamp(variance, min=1e-5)
        delta = torch.sqrt(self.beta * variance)
        return delta - torch.abs(mean - self.target)

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the acquisition function

        Args:
            X (torch.Tensor): Points at which to evaluate.

        Returns:
            torch.Tensor: Value of the acquisition functiona at these points.
        """

        post = self.model.posterior(X)
        samples = self.sampler(post)  # num_samples x batch_shape x q x d_out
        return self.acquisition(self.objective(samples, X)).squeeze(-1)
