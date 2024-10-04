# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Optional

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
