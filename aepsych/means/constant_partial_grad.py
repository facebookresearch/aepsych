#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from gpytorch.means.constant_mean import ConstantMean


class ConstantMeanPartialObsGrad(ConstantMean):
    """A mean function for use with partial gradient observations.

    This follows gpytorch.means.constant_mean_grad and sets the prior mean for
    derivative observations to 0, though unlike that function it allows for
    partial observation of derivatives.

    The final column of input should be an index that is 0 if the observation
    is of f, or i if it is of df/dxi.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        idx = input[..., -1].to(dtype=torch.long) > 0
        mean_fit = super(ConstantMeanPartialObsGrad, self).forward(input[..., ~idx, :])
        sz = mean_fit.shape[:-1] + torch.Size([input.shape[-2]])
        mean = torch.zeros(sz)
        mean[~idx] = mean_fit
        return mean
