# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Any, Optional, Tuple

import numpy as np
import torch
from botorch.posteriors import GPyTorchPosterior
from scipy.special import owens_t
from torch import Tensor
from torch.distributions.normal import Normal

from .base import AEPsychObjective


class ProbitObjective(AEPsychObjective):
    """Probit objective

    Transforms the input through the normal CDF (probit).
    """

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        """Evaluates the objective (normal CDF).

        Args:
            samples (Tensor): GP samples.
            X (Optional[Tensor], optional): ignored, here for compatibility
                with MCAcquisitionObjective.

        Returns:
            Tensor: [description]
        """
        return Normal(loc=0, scale=1).cdf(samples.squeeze(-1))

    def inverse(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        """Evaluates the inverse of the objective (normal PPF).

        Args:
            samples (Tensor): GP samples.
            X (Optional[Tensor], optional): ignored, here for compatibility
                with MCAcquisitionObjective.

        Returns:
            Tensor: [description]
        """
        return Normal(loc=0, scale=1).icdf(samples.squeeze(-1))

    def posterior_transform(
        self, posterior: GPyTorchPosterior, **kwargs: Any
    ) -> Tuple[Tensor, Tensor]:
        # Probability-space mean and variance for Bernoulli-probit models is
        # available in closed form, Proposition 1 in Letham et al. 2022 (AISTATS).
        fmean = posterior.mean.squeeze()
        fvar = posterior.variance.squeeze()
        a_star = fmean / torch.sqrt(1 + fvar)
        pmean = Normal(0, 1).cdf(a_star)
        t_term = torch.tensor(
            owens_t(a_star.numpy(), 1 / np.sqrt(1 + 2 * fvar.numpy())),
            dtype=a_star.dtype,
        )
        pvar = pmean - 2 * t_term - pmean.square()
        return pmean, pvar
