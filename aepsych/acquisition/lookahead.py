#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Optional, Tuple

import numpy as np
import torch
from aepsych.utils import make_scaled_sobol
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.transforms import t_batch_mode_transform
from scipy.stats import norm
from torch import Tensor

from .lookahead_utils import (
    approximate_lookahead_levelset_at_xstar,
    lookahead_levelset_at_xstar,
    lookahead_p_at_xstar,
)


def Hb(p: Tensor):
    """
    Binary entropy.

    Args:
        p: Tensor of probabilities.

    Returns: Binary entropy for each probability.
    """
    epsilon = torch.tensor(np.finfo(float).eps)
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


class LookaheadAcquisitionFunction(AcquisitionFunction):
    def __init__(
        self,
        model: GPyTorchModel,
        target: Optional[float],
        lookahead_type: str = "levelset",
    ) -> None:
        """
        A localized look-ahead acquisition function.

        Args:
            model: The gpytorch model.
            target: Threshold value to target in p-space.
        """
        super().__init__(model=model)
        if lookahead_type == "levelset":
            self.lookahead_fn = lookahead_levelset_at_xstar
            assert target is not None, "Need a target for levelset lookahead!"
            self.gamma = norm.ppf(target)
        elif lookahead_type == "posterior":
            self.lookahead_fn = lookahead_p_at_xstar
            self.gamma = None
        else:
            raise RuntimeError(f"Got unknown lookahead type {lookahead_type}!")


## Global look-ahead acquisitions
class GlobalLookaheadAcquisitionFunction(LookaheadAcquisitionFunction):
    def __init__(
        self,
        model: GPyTorchModel,
        lookahead_type: str = "levelset",
        target: Optional[float] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        query_set_size: Optional[int] = 256,
        Xq: Optional[Tensor] = None,
    ) -> None:
        """
        A global look-ahead acquisition function.

        Args:
            model: The gpytorch model.
            target: Threshold value to target in p-space.
            Xq: (m x d) global reference set.
        """
        super().__init__(model=model, target=target, lookahead_type=lookahead_type)
        self.posterior_transform = posterior_transform
        assert (
            Xq is not None or query_set_size is not None
        ), "Must pass either query set size or a query set!"
        if Xq is not None and query_set_size is not None:
            assert Xq.shape[0] == query_set_size, (
                "If passing both Xq and query_set_size,"
                + "first dim of Xq should be query_set_size, got {Xq.shape[0]} != {query_set_size}"
            )
        if Xq is None:
            # cast to an int in case we got a float from Config, which
            # would raise on make_scaled_sobol
            query_set_size = cast(int, query_set_size)  # make mypy happy
            assert int(query_set_size) == query_set_size  # make sure casting is safe
            # if the asserts above pass and Xq is None, query_set_size is not None so this is safe
            query_set_size = int(query_set_size)  # cast
            Xq = make_scaled_sobol(model.lb, model.ub, query_set_size)
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

        return self.lookahead_fn(
            model=self.model,
            Xstar=X,
            Xq=Xq_batch,
            gamma=self.gamma,
            posterior_transform=self.posterior_transform,
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
    def __init__(
        self,
        model: GPyTorchModel,
        lookahead_type="levelset",
        target: Optional[float] = None,
        query_set_size: Optional[int] = 256,
        Xq: Optional[Tensor] = None,
    ) -> None:
        assert (
            lookahead_type == "levelset"
        ), f"ApproxGlobalSUR only supports lookahead on level set, got {lookahead_type}!"
        super().__init__(
            model=model,
            target=target,
            lookahead_type=lookahead_type,
            query_set_size=query_set_size,
            Xq=Xq,
        )

    def _get_lookahead_posterior(
        self, X: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        Xq_batch = self.Xq.expand(X.shape[0], *self.Xq.shape)

        return approximate_lookahead_levelset_at_xstar(
            model=self.model,
            Xstar=X,
            Xq=Xq_batch,
            gamma=self.gamma,
            posterior_transform=self.posterior_transform,
        )


class EAVC(GlobalLookaheadAcquisitionFunction):
    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        return EAVC_fn(Px, P1, P0, py1)


## Local look-ahead acquisitions
class LocalLookaheadAcquisitionFunction(LookaheadAcquisitionFunction):
    def __init__(
        self,
        model: GPyTorchModel,
        lookahead_type: str = "levelset",
        target: Optional[float] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> None:
        """
        A localized look-ahead acquisition function.

        Args:
            model: The gpytorch model.
            target: Threshold value to target in p-space.
        """

        super().__init__(model=model, target=target, lookahead_type=lookahead_type)
        self.posterior_transform = posterior_transform

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate acquisition function at X.

        Args:
            X: (b x 1 x d) point at which to evalaute acquisition function.

        Returns: (b) tensor of acquisition values.
        """

        Px, P1, P0, py1 = self.lookahead_fn(
            model=self.model,
            Xstar=X,
            Xq=X,
            gamma=self.gamma,
            posterior_transform=self.posterior_transform,
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


class MOCU(GlobalLookaheadAcquisitionFunction):
    """
    MOCU acquisition function given in expr. 4 of:

        Zhao, Guang, et al. "Uncertainty-aware active learning for optimal Bayesian classifier."
        International Conference on Learning Representations (ICLR) 2021.
    """

    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        current_max_query = torch.maximum(Px, 1 - Px)
        # expectation w.r.t. y* of the max of pq
        lookahead_pq1_max = torch.maximum(P1, 1 - P1)
        lookahead_pq0_max = torch.maximum(P0, 1 - P0)
        lookahead_max_query = lookahead_pq1_max * py1 + lookahead_pq0_max * (1 - py1)
        return (lookahead_max_query - current_max_query).mean(-1)


class SMOCU(GlobalLookaheadAcquisitionFunction):
    """
    SMOCU acquisition function given in expr. 11 of:

       Zhao, Guang, et al. "Bayesian active learning by soft mean objective cost of uncertainty."
       International Conference on Artificial Intelligence and Statistics (AISTATS) 2021.
    """

    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        stacked = torch.stack((Px, 1 - Px), dim=-1)
        current_softmax_query = torch.logsumexp(self.k * stacked, dim=-1) / self.k
        # expectation w.r.t. y* of the max of pq
        lookahead_pq1_max = torch.maximum(P1, 1 - P1)
        lookahead_pq0_max = torch.maximum(P0, 1 - P0)
        lookahead_max_query = lookahead_pq1_max * py1 + lookahead_pq0_max * (1 - py1)
        return (lookahead_max_query - current_softmax_query).mean(-1)


class BEMPS(GlobalLookaheadAcquisitionFunction):
    """
    BEMPS acquisition function given in:

        Tan, Wei, et al. "Diversity Enhanced Active Learning with Strictly Proper Scoring Rules."
        Advances in Neural Information Processing Systems 34 (2021).
    """

    def __init__(self, scorefun, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scorefun = scorefun

    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        current_score = self.scorefun(Px)
        lookahead_pq1_score = self.scorefun(P1)
        lookahead_pq0_score = self.scorefun(P0)
        lookahead_expected_score = lookahead_pq1_score * py1 + lookahead_pq0_score * (
            1 - py1
        )
        return (lookahead_expected_score - current_score).mean(-1)
