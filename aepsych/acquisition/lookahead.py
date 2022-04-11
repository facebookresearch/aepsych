#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import numpy as np
import torch
from aepsych.utils import make_scaled_sobol
from botorch.acquisition import AcquisitionFunction
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.transforms import t_batch_mode_transform
from scipy.stats import norm
from torch import Tensor

from .lookahead_utils import approximate_lookahead_at_xstar, lookahead_at_xstar


def Hb(p: Tensor):
    """
    Binary entropy.

    Args:
        p: Tensor of probabilities.

    Returns: Binary entropy for each probability.
    """
    epsilon = np.finfo(float).eps
    p = torch.clamp(p, min=epsilon, max=1 - epsilon)
    return -torch.nan_to_num(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))


def MI_fn(Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
    """
    Average mutual information.
    H(p) - E_y*[H(p | y*)]

    Args:
        Px: (b x m) Level-set posterior before observation
        P1: (b x m) Level-set posterior given observation of 1
        P0: (b x m) Level-set posterior given observation of 0
        py1: (b x 1) Probability of observing 1

    Returns: (b) tensor of mutual information averaged over Xq.
    """
    mi = Hb(Px) - py1 * Hb(P1) - (1 - py1) * Hb(P0)
    return mi.sum(dim=-1)


def ClassErr(p: Tensor) -> Tensor:
    """
    Expected classification error, min(p, 1-p).
    """
    return torch.min(p, 1 - p)


def SUR_fn(Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
    """
    Stepwise uncertainty reduction.

    Expected reduction in expected classification error given observation at Xstar,
    averaged over Xq.

    Args:
        Px: (b x m) Level-set posterior before observation
        P1: (b x m) Level-set posterior given observation of 1
        P0: (b x m) Level-set posterior given observation of 0
        py1: (b x 1) Probability of observing 1

    Returns: (b) tensor of SUR values.
    """
    sur = ClassErr(Px) - py1 * ClassErr(P1) - (1 - py1) * ClassErr(P0)
    return sur.sum(dim=-1)


def EAVC_fn(Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
    """
    Expected absolute value change.

    Expected absolute change in expected level-set volume given observation at Xstar.

    Args:
        Px: (b x m) Level-set posterior before observation
        P1: (b x m) Level-set posterior given observation of 1
        P0: (b x m) Level-set posterior given observation of 0
        py1: (b x 1) Probability of observing 1

    Returns: (b) tensor of EAVC values.
    """
    avc1 = torch.abs((Px - P1).sum(dim=-1))
    avc0 = torch.abs((Px - P0).sum(dim=-1))
    return py1.squeeze(-1) * avc1 + (1 - py1).squeeze(-1) * avc0


## Global look-ahead acquisitions
class GlobalLookaheadAcquisitionFunction(AcquisitionFunction):
    def __init__(
        self,
        model: GPyTorchModel,
        target: float,
        query_set_size: Optional[int] = None,
        Xq: Optional[Tensor] = None,
    ) -> None:
        """
        A global look-ahead acquisition function.

        Args:
            model: The gpytorch model.
            target: Threshold value to target in p-space.
            Xq: (m x d) global reference set.
        """
        super().__init__(model=model)
        assert (
            Xq is not None or query_set_size is not None
        ), "Must pass either query set size or a query set!"
        if Xq is not None and query_set_size is not None:
            assert Xq.shape[0] == query_set_size, (
                "If passing both Xq and query_set_size,"
                + "first dim of Xq should be query_set_size, got {Xq.shape[0]} != {query_set_size}"
            )
        self.gamma = norm.ppf(target)
        Xq = (
            Xq
            if Xq is not None
            else make_scaled_sobol(model.lb, model.ub, query_set_size)
        )
        self.register_buffer("Xq", Xq)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate acquisition function at X.

        Args:
            X: (b x 1 x d) point at which to evalaute acquisition function.

        Returns: (b) tensor of acquisition values.
        """
        Px, P1, P0, py1 = self._get_lookahead_posterior(X)
        return self._compute_acqf(Px, P1, P0, py1)

    def _get_lookahead_posterior(
        self, X: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        Xq_batch = self.Xq.expand(X.shape[0], *self.Xq.shape)
        return lookahead_at_xstar(
            model=self.model, Xstar=X, Xq=Xq_batch, gamma=self.gamma
        )

    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        raise NotImplementedError


class GlobalMI(GlobalLookaheadAcquisitionFunction):
    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        return MI_fn(Px, P1, P0, py1)


class GlobalSUR(GlobalLookaheadAcquisitionFunction):
    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        return SUR_fn(Px, P1, P0, py1)


class ApproxGlobalSUR(GlobalSUR):
    def _get_lookahead_posterior(
        self, X: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        Xq_batch = self.Xq.expand(X.shape[0], *self.Xq.shape)
        return approximate_lookahead_at_xstar(
            model=self.model, Xstar=X, Xq=Xq_batch, gamma=self.gamma
        )


class EAVC(GlobalLookaheadAcquisitionFunction):
    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        return EAVC_fn(Px, P1, P0, py1)


## Local look-ahead acquisitions


class LocalLookaheadAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model: GPyTorchModel, target: float) -> None:
        """
        A localized look-ahead acquisition function.

        Args:
            model: The gpytorch model.
            target: Threshold value to target in p-space.
        """
        super().__init__(model=model)
        self.gamma = norm.ppf(target)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate acquisition function at X.

        Args:
            X: (b x 1 x d) point at which to evalaute acquisition function.

        Returns: (b) tensor of acquisition values.
        """
        Px, P1, P0, py1 = lookahead_at_xstar(
            model=self.model, Xstar=X, Xq=X, gamma=self.gamma
        )  # Return shape here has m=1.
        return self._compute_acqf(Px, P1, P0, py1)

    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        raise NotImplementedError


class LocalMI(LocalLookaheadAcquisitionFunction):
    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        return MI_fn(Px, P1, P0, py1)


class LocalSUR(LocalLookaheadAcquisitionFunction):
    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        return SUR_fn(Px, P1, P0, py1)
