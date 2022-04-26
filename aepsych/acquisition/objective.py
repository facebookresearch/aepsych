#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional
import torch
from botorch.acquisition.objective import MCAcquisitionObjective
from torch import Tensor
from torch.distributions.normal import Normal


class ProbitObjective(MCAcquisitionObjective):
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


class FloorLinkObjective(MCAcquisitionObjective):
    """
    Wrapper for objectives to add a floor, when
    the probability is known not to go below it.
    """

    def __init__(self, floor=0.5):
        self.floor = floor
        super().__init__()

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        """Evaluates the objective for input x and floor f

        Args:
            samples (Tensor): GP samples.
            X (Optional[Tensor], optional): ignored, here for compatibility
                with MCAcquisitionObjective.

        Returns:
            Tensor: outcome probability.
        """
        return self.link(samples.squeeze(-1)) * (1 - self.floor) + self.floor

    def link(self, samples):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        floor = config.getfloat(cls.__name__, "floor")
        return cls(floor=floor)


class FloorLogitObjective(FloorLinkObjective):
    """
    Logistic sigmoid (aka expit, aka logistic CDF),
    but with a floor so that its output is
    between floor and 1.0.
    """

    def link(self, samples):
        return torch.special.expit(samples)


class FloorGumbelObjective(FloorLinkObjective):
    """
    Gumbel CDF but with a floor so that its output
    is between floor and 1.0. Note that this is not
    the standard Gumbel distribution, but rather the
    left-skewed Gumbel that arises as the log of the Weibull
    distribution, e.g. Treutwein 1995, doi:10.1016/0042-6989(95)00016-X.
    """

    def link(self, samples):
        return torch.nan_to_num(
            1 - torch.exp(-torch.exp(samples)), posinf=1.0, neginf=0.0
        )


class FloorProbitObjective(FloorLinkObjective):
    """
    Probit (aka Gaussian CDF), but with a floor
    so that its output is between floor and 1.0.
    """

    def link(self, samples):
        return Normal(0, 1).cdf(samples)
