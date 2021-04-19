#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
Mutual Information Acquisition functions

With latent function F and X a hypothetical observation at a new point,
I(F; X) = I(X; F) = H(X) - H(X |F),
H(X |F ) = E_{f} (H(X |F =f )
i.e., we take the posterior entropy of the (Bernoulli) observation X given the
current model posterior and subtract the conditional entropy on F, that being
the mean entropy over the posterior for F. This is equivalent to the BALD
acquisition function in: Gardner et al. 2015, Psychophysical detection
testing with Bayesian active learning.
"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from aepsych.acquisition.monotonic_rejection import MonotonicMCAcquisition
from torch import Tensor
from torch.distributions.bernoulli import Bernoulli


def bald_acq(obj_samples):
    # the output of objective is of shape num_samples x batch_shape x d_out
    mean_p = obj_samples.mean(dim=0)
    posterior_entropies = Bernoulli(mean_p).entropy().squeeze(-1)
    sample_entropies = Bernoulli(obj_samples).entropy()
    conditional_entropies = sample_entropies.mean(dim=0).squeeze(-1)

    return posterior_entropies - conditional_entropies


class BernoulliMCMutualInformation(MCAcquisitionFunction):
    """Mutual Information acquisition function for a bernoulli outcome.
    Given a model and an objective link function, calculate the mutual
    information of a  trial at a new point and the distribution on the
    latent function.
    objective here should give values in (0, 1) (e.g. logit or probit)/
    """

    def __init__(
        self,
        model: Model,
        objective: MCAcquisitionObjective,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        r"""Single Bernoulli mutual information for active learning

        Args:
            model: A fitted model.
            objective: An MCAcquisitionObjective representing the link function
                (e.g., logistic or probit.)
            sampler: The sampler used for drawing MC samples.
        """
        if sampler is None:
            sampler = SobolQMCNormalSampler(num_samples=1024, collapse_batch_dims=True)
        super().__init__(
            model=model, sampler=sampler, objective=objective, X_pending=None
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate mutual information on the candidate set `X`.

        Args:
            X: A `batch_size x q x d`-dim Tensor.
        Returns:
            Tensor of shape `batch_size x q` representing the mutual
            information of a hypothetical trial at X that active
            learning hopes to maximize
        """
        post = self.model.posterior(X)
        samples = self.sampler(post)

        return self.acquisition(self.objective(samples))

    def acquisition(self, obj_samples: torch.Tensor) -> torch.Tensor:
        # RejectionSampler drops the final dim so we reaugment it
        # here for compatibility with non-Monotonic MCAcquisition
        if len(obj_samples.shape) == 2:
            obj_samples = obj_samples[..., None]
        return bald_acq(obj_samples)


class MonotonicBernoulliMCMutualInformation(MonotonicMCAcquisition):
    def acquisition(self, obj_samples: torch.Tensor) -> torch.Tensor:
        # RejectionSampler drops the final dim so we reaugment it
        # here for compatibility with non-Monotonic MCAcquisition
        if len(obj_samples.shape) == 2:
            obj_samples = obj_samples[..., None]
        return bald_acq(obj_samples)
