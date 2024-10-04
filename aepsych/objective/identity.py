# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Any, Optional, Tuple

import torch
from botorch.posteriors import GPyTorchPosterior
from torch import Tensor

from .base import AEPsychObjective


class IdentityObjective(AEPsychObjective):
    r"""Trivial objective extracting the last dimension.

    Example:
        >>> identity_objective = IdentityMCObjective()
        >>> samples = sampler(posterior)
        >>> objective = identity_objective(samples)
    """

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        return samples.squeeze(-1)

    def posterior_transform(
        self, posterior: GPyTorchPosterior, **kwargs: Any
    ) -> Tuple[Tensor, Tensor]:
        """Evaluates the objective on a posterior, returning mean and variance.
        This base implementation uses Monte Carlo sampling, but some objectives may have analytic solutions.

        Args:
            posterior (botorch.posteriors.GPyTorchPosterior): A posterior evaluated at some points.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed posterior mean and variance.
        """

        samps = kwargs.get("samples", 10000)
        fmean = posterior.mean.squeeze()
        fvar = posterior.variance.squeeze()
        return fmean, fvar
