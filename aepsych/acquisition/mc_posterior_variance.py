#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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


def balv_acq(obj_samps: torch.Tensor) -> torch.Tensor:
    """Evaluate BALV (posterior variance) on a set of objective samples.

    Args:
        obj_samps (torch.Tensor): Samples from the GP, transformed by the objective.
            Should be samples x batch_shape.

    Returns:
        torch.Tensor: Acquisition function value.
    """

    # the output of objective is of shape num_samples x batch_shape x d_out
    # objective should project the last dimension to 1d,
    # so incoming should be samples x batch_shape, we take var in samp dim
    return obj_samps.var(dim=0).squeeze(-1)


class MCPosteriorVariance(MCAcquisitionFunction):
    r"""Posterior variance, computed using samples so we can use objective/transform"""

    def __init__(
        self,
        model: Model,
        objective: Optional[MCAcquisitionObjective] = None,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        r"""Posterior Variance of Link Function

        Args:
            model (Model): A fitted model.
            objective (MCAcquisitionObjective optional): An MCAcquisitionObjective representing the link function
                (e.g., logistic or probit.) applied on the difference of (usually 1-d)
                two samples. Can be implemented via GenericMCObjective. Defaults tp ProbitObjective.
            sampler (MCSampler, optional): The sampler used for drawing MC samples. Defaults to SobolQMCNormalSampler.
        """
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        if objective is None:
            objective = ProbitObjective()
        super().__init__(model=model, sampler=sampler, objective=None, X_pending=None)
        self.objective = objective

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate MCPosteriorVariance on the candidate set `X`.

        Args:
            X (torch.Tensor): A `batch_size x q x d`-dim Tensor

        Returns:
            torch.Tensor: Posterior variance of link function at X that active learning
            hopes to maximize
        """
        # the output is of shape batch_shape x q x d_out
        post = self.model.posterior(X)
        samples = self.sampler(post)  # num_samples x batch_shape x q x d_out

        return self.acquisition(self.objective(samples, X))

    def acquisition(self, obj_samples: torch.Tensor) -> torch.Tensor:
        """Evaluate the acquisition based on objective samples.

        Args:
            obj_samples (torch.Tensor): Samples from the GP, transformed by the objective.
                Should be samples x batch_shape.

        Returns:
            torch.Tensor: Acquisition function at the sampled values.
        """
        # RejectionSampler drops the final dim so we reaugment it
        # here for compatibility with non-Monotonic MCAcquisition
        if len(obj_samples.shape) == 2:
            obj_samples = obj_samples[..., None]
        return balv_acq(obj_samples)


@acqf_input_constructor(MCPosteriorVariance)
def construct_inputs(
    model: Model,
    training_data: None,
    objective: Optional[MCAcquisitionObjective] = None,
    sampler: Optional[MCSampler] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Constructs the input dictionary for initializing the MCPosteriorVariance acquisition function.

    Args:
        model (Model): The fitted model to be used.
        training_data (None): Placeholder for compatibility; not used in this function.
        objective (MCAcquisitionObjective, optional): Objective function for transforming samples (e.g., logistic or probit).
        sampler (MCSampler, optional): Sampler for Monte Carlo sampling; defaults to SobolQMCNormalSampler if not provided.

    Returns:
        Dict[str, Any]: Dictionary of constructed inputs for the MCPosteriorVariance acquisition function.
    """
    return {
        "model": model,
        "objective": objective,
        "sampler": sampler,
    }


class MonotonicMCPosteriorVariance(MonotonicMCAcquisition):
    def acquisition(self, obj_samples: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the acquisition function value for monotonic posterior variance.

        Args:
            obj_samples (torch.Tensor): Samples from the GP, transformed by the objective.
                Should have shape samples x batch_shape.

        Returns:
            torch.Tensor: The BALV acquisition function value, representing the posterior variance
            calculated over the sample dimension.
        """

        return balv_acq(obj_samples)
