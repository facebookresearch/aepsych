#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional

from botorch.acquisition.objective import MCAcquisitionObjective
from torch import Tensor
from torch.distributions.normal import Normal


class ProbitObjective(MCAcquisitionObjective):
    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        return Normal(loc=0, scale=1).cdf(samples.squeeze(-1))
