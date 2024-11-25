#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.posteriors import Posterior
from botorch.sampling.base import MCSampler


class RejectionSampler(MCSampler):
    """
    Samples from a posterior subject to the constraint that samples in constrained_idx
    should be >= 0.

    If not enough feasible samples are generated, will return the least violating
    samples.
    """

    def __init__(
        self,
        num_samples: int,
        num_rejection_samples: int,
        constrained_idx: torch.Tensor,
    ):
        """Initialize RejectionSampler

        Args:
            num_samples (int): Number of samples to return. Note that if fewer samples
                than this number are positive in the required dimension, the remaining
                samples returned will be the "least violating", i.e. closest to 0.
            num_rejection_samples (int): Number of samples to draw before rejecting.
            constrained_idx (torch.Tensor): Indices of input dimensions that should be
                constrained positive.
        """
        self.num_samples = num_samples
        self.num_rejection_samples = num_rejection_samples
        self.constrained_idx = constrained_idx
        super().__init__(sample_shape=torch.Size([num_samples]))

    def forward(self, posterior: Posterior) -> torch.Tensor:
        """Run the rejection sampler.

        Args:
            posterior (Posterior): The unconstrained GP posterior object
                to perform rejection samples on.

        Returns:
            torch.Tensor: Kept samples.
        """
        samples = posterior.rsample(
            sample_shape=torch.Size([self.num_rejection_samples])
        )
        assert (
            samples.shape[-1] == 1
        ), "Batches not supported"  # TODO T68656582 handle batches later
        constrained_samps = samples[:, self.constrained_idx, 0]
        valid = (constrained_samps >= 0).all(dim=1)
        if valid.sum() < self.num_samples:
            worst_violation = constrained_samps.min(dim=1)[0]
            keep = torch.argsort(worst_violation, descending=True)[: self.num_samples]
        else:
            keep = torch.where(valid)[0][: self.num_samples]
        return samples[keep, :, :]
