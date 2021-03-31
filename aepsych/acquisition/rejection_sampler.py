#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

import torch
from botorch.posteriors import Posterior
from botorch.sampling.samplers import MCSampler
from torch import Tensor


class RejectionSampler(MCSampler):
    """
    Samples from a posterior subject to the constraint that samples in constrained_idx
    should be >= 0.

    If not enough feasible samples are generated, will return the least violating
    samples.
    """

    def __init__(
        self, num_samples: int, num_rejection_samples: int, constrained_idx: Tensor
    ):
        self.num_samples = num_samples
        self.num_rejection_samples = num_rejection_samples
        self.constrained_idx = constrained_idx
        self._sample_shape = torch.Size([num_samples])
        super().__init__()

    def _get_base_sample_shape(self, posterior: Posterior) -> torch.Size:
        return torch.Size([])

    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) -> None:
        self.base_samples = None

    def forward(self, posterior: Posterior) -> Tensor:
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
