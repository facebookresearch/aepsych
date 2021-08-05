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
