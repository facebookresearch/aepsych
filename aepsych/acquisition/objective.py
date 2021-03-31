#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import Optional

from botorch.acquisition.objective import MCAcquisitionObjective
from torch import Tensor
from torch.distributions.normal import Normal


class ProbitObjective(MCAcquisitionObjective):
    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        return Normal(loc=0, scale=1).cdf(samples.squeeze(-1))
