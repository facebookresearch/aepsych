#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch
from gpytorch.kernels.rbf_kernel_grad import RBFKernelGrad


class RBFKernelPartialObsGrad(RBFKernelGrad):
    """An RBF kernel over observations of f, and partial/non-overlapping
    observations of the gradient of f.

    gpytorch.kernels.rbf_kernel_grad assumes a block structure where every
    partial derivative is observed at the same set of points at which x is
    observed. This generalizes that by allowing f and any subset of the
    derivatives of f to be observed at different sets of points.

    The final column of x1 and x2 needs to be an index that identifies what is
    observed at that point. It should be 0 if this observation is of f, and i
    if it is of df/dxi.
    """

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params: Any
    ) -> torch.Tensor:
        # Extract grad index from each
        grad_idx1 = x1[..., -1].to(dtype=torch.long)
        grad_idx2 = x2[..., -1].to(dtype=torch.long)
        K = super().forward(x1[..., :-1], x2[..., :-1], diag=diag, **params)
        # Compute which elements to return
        n1 = x1.shape[-2]
        n2 = x2.shape[-2]
        d = x1.shape[-1] - 1
        p1 = [(i * (d + 1)) + int(grad_idx1[i]) for i in range(n1)]
        p2 = [(i * (d + 1)) + int(grad_idx2[i]) for i in range(n2)]
        if not diag:
            return K[..., p1, :][..., p2]
        else:
            return K[..., p1]

    def num_outputs_per_input(self, x1: torch.Tensor, x2: torch.Tensor) -> int:
        return 1
