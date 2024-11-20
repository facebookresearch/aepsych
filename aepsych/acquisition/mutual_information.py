#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from aepsych.acquisition.monotonic_rejection import MonotonicMCAcquisition
from aepsych.acquisition.objective import ProbitObjective
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from torch.distributions.bernoulli import Bernoulli


def bald_acq(obj_samples: torch.Tensor) -> torch.Tensor:
    """Evaluate Mutual Information acquisition function.

    With latent function F and X a hypothetical observation at a new point,
    I(F; X) = I(X; F) = H(X) - H(X |F),
    H(X |F ) = E_{f} (H(X |F =f )
    i.e., we take the posterior entropy of the (Bernoulli) observation X given the
    current model posterior and subtract the conditional entropy on F, that being
    the mean entropy over the posterior for F. This is equivalent to the BALD
    acquisition function in Houlsby et al. NeurIPS 2012.

    Args:
        obj_samples (torch.Tensor): Objective samples from the GP, of
            shape num_samples x batch_shape x d_out

    Returns:
        torch.Tensor: Value of acquisition at samples.
    """
    mean_p = obj_samples.mean(dim=0)
    posterior_entropies = Bernoulli(mean_p).entropy().squeeze(-1)
    sample_entropies = Bernoulli(obj_samples).entropy()
    conditional_entropies = sample_entropies.mean(dim=0).squeeze(-1)

    return posterior_entropies - conditional_entropies


class BernoulliMCMutualInformation(MCAcquisitionFunction):
    """Mutual Information acquisition function for a bernoulli outcome.

    Given a model and an objective link function, calculate the mutual
    information of a trial at a new point and the distribution on the
    latent function.

    Objective here should give values in (0, 1) (e.g. logit or probit).
    """

    def __init__(
        self,
        model: Model,
        objective: MCAcquisitionObjective = None,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        r"""Single Bernoulli mutual information for active learning

        Args:
            model (Model): A fitted model.
            objective (MCAcquisitionObjective): An MCAcquisitionObjective representing the link function
                (e.g., logistic or probit)
            sampler (MCSampler, optional): The sampler used for drawing MC samples.
        """
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
        if objective is None:
            objective = ProbitObjective()
        super().__init__(
            model=model, sampler=sampler, objective=objective, X_pending=None
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate mutual information on the candidate set `X`.

        Args:
            X (Tensor): A `batch_size x q x d`-dim Tensor.
        Returns:
            Tensor of shape `batch_size x q` representing the mutual
            information of a hypothetical trial at X that active
            learning hopes to maximize.
        """
        post = self.model.posterior(X)
        samples = self.sampler(post)

        return self.acquisition(self.objective(samples, X))

    def acquisition(self, obj_samples: torch.Tensor) -> torch.Tensor:
        """Evaluate the acquisition function value based on samples.

        Args:
            obj_samples (torch.Tensor): Samples from the model, transformed through the objective.

        Returns:
            torch.Tensor: value of the acquisition function (BALD) at the input samples.
        """
        # RejectionSampler drops the final dim so we reaugment it
        # here for compatibility with non-Monotonic MCAcquisition
        if len(obj_samples.shape) == 2:
            obj_samples = obj_samples[..., None]
        return bald_acq(obj_samples)


@acqf_input_constructor(BernoulliMCMutualInformation)
def construct_inputs_mi(
    model: Model,
    training_data: None,
    objective: Optional[MCAcquisitionObjective] = None,
    sampler: Optional[MCSampler] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Constructs the input dictionary for initializing the BernoulliMCMutualInformation acquisition function.

    Args:
        model (Model): The fitted model to use.
        training_data (None): Placeholder for compatibility; not used in this function.
        objective (MCAcquisitionObjective, optional): Objective function for transforming samples (e.g., logit or probit).
        sampler (MCSampler, optional): Sampler for Monte Carlo sampling; defaults to SobolQMCNormalSampler if not provided.

    Returns:
        Dict[str, Any]: Dictionary of constructed inputs for the BernoulliMCMutualInformation acquisition function.
    """

    return {
        "model": model,
        "objective": objective,
        "sampler": sampler,
    }


class MonotonicBernoulliMCMutualInformation(MonotonicMCAcquisition):
    def acquisition(self, obj_samples: torch.Tensor) -> torch.Tensor:
        """Evaluate the acquisition function value based on samples.

        Args:
            obj_samples (torch.Tensor): Samples from the model, transformed through the objective.

        Returns:
            torch.Tensor: value of the acquisition function (BALD) at the input samples.
        """
        # TODO this is identical to nono-monotonic BALV acquisition with a different
        # base class mixin, consider redesigning?
        # RejectionSampler drops the final dim so we reaugment it
        # here for compatibility with non-Monotonic MCAcquisition
        if len(obj_samples.shape) == 2:
            obj_samples = obj_samples[..., None]
        return bald_acq(obj_samples)
