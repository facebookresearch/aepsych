#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
from aepsych.acquisition.objective import ProbitObjective
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
    MCAcquisitionObjective,
    MCSampler,
)
from botorch.acquisition.objective import ScalarizedObjective
from botorch.models.model import Model
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class LevelSetEstimation(AnalyticAcquisitionFunction):
    r"""Level set estimation.

    Analytic level set estimation as in Gotovos et al. 2013 IJCAI.

    `LSE(x) = sqrt(beta) * sigma(x)- |mu(x) - h|`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> LSE = LevelSetEstimation(model, target=0.5, beta=0.2)
        >>> lse = LSE(test_X)
    """

    def __init__(
        self,
        model: Model,
        target: Union[float, Tensor],
        beta: Union[float, Tensor],
        objective: Optional[ScalarizedObjective] = None,
    ) -> None:
        r"""Single-outcome Level Set Estimation

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            objective: A ScalarizedObjective (optional).
        """
        super().__init__(model=model, objective=objective)
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

        if not torch.is_tensor(target):
            target = torch.tensor(target)
        self.register_buffer("target", target)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the LSE on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of LSE values at the given
            design points `X`.
        """
        self.beta: torch.Tensor = self.beta.to(X)
        self.target: torch.Tensor = self.target.to(X)
        posterior = self._get_posterior(X=X)
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        variance = posterior.variance.view(batch_shape)
        delta = (self.beta.expand_as(mean) * variance).sqrt()

        return delta - torch.abs(mean - self.target)


class MCLevelSetEstimation(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        target: Union[float, Tensor],
        beta: Union[float, Tensor],
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
            sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)
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
        return self.acquisition(self.objective(samples)).squeeze(-1)
