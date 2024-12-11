#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, cast, Dict, Literal, Optional, Tuple

import numpy as np
import torch
from aepsych.utils import make_scaled_sobol
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import acqf_input_constructor
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


def Hb(p: torch.Tensor) -> torch.Tensor:
    """
    Binary entropy.

    Args:
        p (torch.Tensor): torch.Tensor of probabilities.

    Returns: Binary entropy for each probability.
    """
    epsilon = torch.tensor(np.finfo(float).eps).to(p)
    p = torch.clamp(p, min=epsilon, max=1 - epsilon)
    return -torch.nan_to_num(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))


def MI_fn(
    Px: torch.Tensor, P1: torch.Tensor, P0: torch.Tensor, py1: torch.Tensor
) -> torch.Tensor:
    """
    Average mutual information.
    H(p) - E_y*[H(p | y*)]

    Args:
        Px (torch.Tensor): (b x m) Level-set posterior before observation
        P1 (torch.Tensor): (b x m) Level-set posterior given observation of 1
        P0 (torch.Tensor): (b x m) Level-set posterior given observation of 0
        py1 (torch.Tensor): (b x 1) Probability of observing 1

    Returns: (b) torch.tensor of mutual information averaged over Xq.
    """
    mi = Hb(Px) - py1 * Hb(P1) - (1 - py1) * Hb(P0)
    return mi.sum(dim=-1)


def ClassErr(p: torch.Tensor) -> torch.Tensor:
    """
    Expected classification error, min(p, 1-p).

    Args:
        p (torch.Tensor): torch.Tensor of predicted probabilities.

    Returns:
        torch.Tensor: Expected classification error for each probability, computed as min(p, 1 - p).
    """
    return torch.min(p, 1 - p)


def SUR_fn(
    Px: torch.Tensor, P1: torch.Tensor, P0: torch.Tensor, py1: torch.Tensor
) -> torch.Tensor:
    """
    Stepwise uncertainty reduction.

    Expected reduction in expected classification error given observation at Xstar,
    averaged over Xq.

    Args:
        Px (torch.Tensor): (b x m) Level-set posterior before observation
        P1 (torch.Tensor): (b x m) Level-set posterior given observation of 1
        P0 (torch.Tensor): (b x m) Level-set posterior given observation of 0
        py1 (torch.Tensor): (b x 1) Probability of observing 1

    Returns:
        (b) torch.Tensor of SUR values.
    """
    P1 = P1.to(Px)
    py1 = py1.to(Px)
    sur = ClassErr(Px) - py1 * ClassErr(P1) - (1 - py1) * ClassErr(P0)
    return sur.sum(dim=-1)


def EAVC_fn(
    Px: torch.Tensor, P1: torch.Tensor, P0: torch.Tensor, py1: torch.Tensor
) -> torch.Tensor:
    """
    Expected absolute value change.

    Expected absolute change in expected level-set volume given observation at Xstar.

    Args:
        Px (torch.Tensor): (b x m) Level-set posterior before observation
        P1 (torch.Tensor): (b x m) Level-set posterior given observation of 1
        P0 (torch.Tensor): (b x m) Level-set posterior given observation of 0
        py1 (torch.Tensor): (b x 1) Probability of observing 1

    Returns:
        (b) torch.Tensor of EAVC values.
    """
    avc1 = torch.abs((Px - P1).sum(dim=-1))
    avc0 = torch.abs((Px - P0).sum(dim=-1))
    return py1.squeeze(-1) * avc1 + (1 - py1).squeeze(-1) * avc0


class LookaheadAcquisitionFunction(AcquisitionFunction):
    def __init__(
        self,
        model: GPyTorchModel,
        target: Optional[float],
        lookahead_type: Literal["levelset", "posterior"] = "levelset",
    ) -> None:
        """
        A localized look-ahead acquisition function.

        Args:
            model (GPyTorchModel): The gpytorch model to use.
            target (float, optional): Threshold value to target in p-space.
            lookahead_type (Literal["levelset", "posterior"]): The type of look-ahead to perform (default is "levelset").
                - If the lookahead_type is "levelset", the acqf will consider the posterior probability that a point is above or below the target level set.
                - If the lookahead_type is "posterior", the acqf will consider the posterior probability that a point will be detected or not.
        """
        super().__init__(model=model)
        if lookahead_type == "levelset":
            self.lookahead_fn = lookahead_levelset_at_xstar
            assert target is not None, "Need a target for levelset lookahead!"
            self.gamma = norm.ppf(target)
        elif lookahead_type == "posterior":
            self.lookahead_fn = lookahead_p_at_xstar  # type: ignore
            self.gamma = None
        else:
            raise RuntimeError(f"Got unknown lookahead type {lookahead_type}!")


## Local look-ahead acquisitions
class LocalLookaheadAcquisitionFunction(LookaheadAcquisitionFunction):
    def __init__(
        self,
        model: GPyTorchModel,
        lookahead_type: Literal["levelset", "posterior"] = "levelset",
        target: Optional[float] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> None:
        """
        A localized look-ahead acquisition function.

        Args:
            model (GPyTorchModel): The gpytorch model to use.
            lookahead_type (Literal["levelset", "posterior"]): The type of look-ahead to perform (default is "levelset").
                - If the lookahead_type is "levelset", the acqf will consider the posterior probability that a point is above or below the target level set.
                - If the lookahead_type is "posterior", the acqf will consider the posterior probability that a point will be detected or not.
            target (float, optional): Threshold value to target in p-space.
            posterior_transform (PosteriorTransform, optional): Optional transformation to apply to the posterior. Default: None.
        """

        super().__init__(model=model, target=target, lookahead_type=lookahead_type)
        self.posterior_transform = posterior_transform

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate acquisition function at X.

        Args:
            X (torch.Tensor): (b x 1 x d) point at which to evalaute acquisition function.

        Returns:
            (b) torch.Tensor of acquisition values.
        """

        Px, P1, P0, py1 = self.lookahead_fn(
            model=self.model,
            Xstar=X,
            Xq=X,
            gamma=self.gamma,
            posterior_transform=self.posterior_transform,
        )  # Return shape here has m=1.
        return self._compute_acqf(Px, P1, P0, py1)

    def _compute_acqf(
        self, Px: torch.Tensor, P1: torch.Tensor, P0: torch.Tensor, py1: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class LocalMI(LocalLookaheadAcquisitionFunction):
    def _compute_acqf(
        self, Px: torch.Tensor, P1: torch.Tensor, P0: torch.Tensor, py1: torch.Tensor
    ) -> torch.Tensor:
        return MI_fn(Px, P1, P0, py1)


class LocalSUR(LocalLookaheadAcquisitionFunction):
    def _compute_acqf(
        self, Px: torch.Tensor, P1: torch.Tensor, P0: torch.Tensor, py1: torch.Tensor
    ) -> torch.Tensor:
        return SUR_fn(Px, P1, P0, py1)


@acqf_input_constructor(LocalMI, LocalSUR)
def construct_inputs_local_lookahead(
    model: GPyTorchModel,
    training_data: None,
    lookahead_type: Literal["levelset", "posterior"] = "levelset",
    target: Optional[float] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Constructs the input dictionary for initializing local lookahead acquisition functions.

    Args:
        model (GPyTorchModel): The gpytorch model to use.
        training_data (None): Placeholder for compatibility; not used in this function.
        lookahead_type (Literal["levelset", "posterior"]): Type of look-ahead to perform. Default is "levelset".
            - If the lookahead_type is "levelset", the acqf will consider the posterior probability that a point is above or below the target level set.
            - If the lookahead_type is "posterior", the acqf will consider the posterior probability that a point will be detected or not.
        target (float, optional): Target threshold value in probability space. Default is None.
        posterior_transform (PosteriorTransform, optional): Optional transformation to apply to the posterior. Default is None.

    Returns:
        Dict[str, Any]: Dictionary of constructed inputs for local lookahead acquisition functions.
    """
    return {
        "model": model,
        "lookahead_type": lookahead_type,
        "target": target,
        "posterior_transform": posterior_transform,
    }


## Global look-ahead acquisitions
class GlobalLookaheadAcquisitionFunction(LookaheadAcquisitionFunction):
    def __init__(
        self,
        lb: Tensor,
        ub: Tensor,
        model: GPyTorchModel,
        lookahead_type: Literal["levelset", "posterior"] = "levelset",
        target: Optional[float] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        query_set_size: Optional[int] = 256,
        Xq: Optional[torch.Tensor] = None,
    ) -> None:
        """
        A global look-ahead acquisition function.

        Args:
            lb (Tensor): Lower bounds of the input space, used to generate the query set (Xq).
            ub (Tensor): Upper bounds of the input space, used to generate the query set (Xq).
            model (GPyTorchModel): The gpytorch model.
            lookahead_type (Literal["levelset", "posterior"]): The type of look-ahead to perform (default is "levelset").
                - If the lookahead_type is "levelset", the acqf will consider the posterior probability that a point is above or below the target level set.
                - If the lookahead_type is "posterior", the acqf will consider the posterior probability that a point will be detected or not.
            target (float, optional): Threshold value to target in p-space.
            posterior_transform (PosteriorTransform, optional): Posterior transform to use. Defaults to None.
            query_set_size (int, optional): Size of the query set. Defaults to 256.
            Xq (Tensor, optional): (m x d) global reference set. Defaults to None.
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
            Xq = make_scaled_sobol(lb, ub, query_set_size)
        self.register_buffer("Xq", Xq)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate acquisition function at X.

        Args:
            X (torch.Tensor): (b x 1 x d) point at which to evalaute acquisition function.

        Returns:
            (b) torch.Tensor of acquisition values.
        """
        Px, P1, P0, py1 = self._get_lookahead_posterior(X)
        return self._compute_acqf(Px, P1, P0, py1)

    def _get_lookahead_posterior(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Xq_batch = self.Xq.expand(X.shape[0], *self.Xq.shape)

        return self.lookahead_fn(
            model=self.model,
            Xstar=X,
            Xq=Xq_batch,
            gamma=self.gamma,
            posterior_transform=self.posterior_transform,
        )

    def _compute_acqf(
        self, Px: torch.Tensor, P1: torch.Tensor, P0: torch.Tensor, py1: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class GlobalMI(GlobalLookaheadAcquisitionFunction):
    def _compute_acqf(
        self, Px: torch.Tensor, P1: torch.Tensor, P0: torch.Tensor, py1: torch.Tensor
    ) -> torch.Tensor:
        return MI_fn(Px, P1, P0, py1)


class GlobalSUR(GlobalLookaheadAcquisitionFunction):
    def _compute_acqf(
        self, Px: torch.Tensor, P1: torch.Tensor, P0: torch.Tensor, py1: torch.Tensor
    ) -> torch.Tensor:
        return SUR_fn(Px, P1, P0, py1)


class ApproxGlobalSUR(GlobalSUR):
    def __init__(
        self,
        lb: Tensor,
        ub: Tensor,
        model: GPyTorchModel,
        lookahead_type: Literal["levelset", "poserior"] = "levelset",
        target: Optional[float] = None,
        query_set_size: Optional[int] = 256,
        Xq: Optional[torch.Tensor] = None,
    ) -> None:
        """
        An approximate global look-ahead acquisition function.
        Args:

            model (GPyTorchModel): The gpytorch model to use.
            lookahead_type (Literal["levelset", "posterior"]): The type of look-ahead to perform (default is "levelset").
                - If the lookahead_type is "levelset", the acqf will consider the posterior probability that a point is above or below the target level set.
                - If the lookahead_type is "posterior", the acqf will consider the posterior probability that a point will be detected or not.
            target (float, optional): Threshold value to target in p-space.
            query_set_size (int, optional): Number of points in the query set.
            Xq (torch.Tensor, optional): (m x d) global reference set.
        """
        assert (
            lookahead_type == "levelset"
        ), f"ApproxGlobalSUR only supports lookahead on level set, got {lookahead_type}!"
        super().__init__(
            lb=lb,
            ub=ub,
            model=model,
            target=target,
            lookahead_type=lookahead_type,
            query_set_size=query_set_size,
            Xq=Xq,
        )

    def _get_lookahead_posterior(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the look-ahead posterior distribution for given points.
        Args:
            X (torch.Tensor): Input torch.Tensor representing the points at which to evaluate the look-ahead posterior.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing tensors corresponding to the posterior's computed values.
        """
        Xq_batch = self.Xq.expand(X.shape[0], *self.Xq.shape)

        return approximate_lookahead_levelset_at_xstar(
            model=self.model,
            Xstar=X,
            Xq=Xq_batch,
            gamma=self.gamma,
            posterior_transform=self.posterior_transform,
        )


class EAVC(GlobalLookaheadAcquisitionFunction):
    def _compute_acqf(
        self, Px: torch.Tensor, P1: torch.Tensor, P0: torch.Tensor, py1: torch.Tensor
    ) -> torch.Tensor:
        return EAVC_fn(Px, P1, P0, py1)


class MOCU(GlobalLookaheadAcquisitionFunction):
    """
    MOCU acquisition function given in expr. 4 of:

        Zhao, Guang, et al. "Uncertainty-aware active learning for optimal Bayesian classifier."
        International Conference on Learning Representations (ICLR) 2021.
    """

    def _compute_acqf(
        self, Px: torch.Tensor, P1: torch.Tensor, P0: torch.Tensor, py1: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the MOCU (Measure of Class Uncertainty) acquisition function.

        Args:
            Px (torch.Tensor): Tensor representing the current probability of class prediction.
            P1 (torch.Tensor): Tensor of look-ahead predictions for the positive class.
            P0 (torch.Tensor): Tensor of look-ahead predictions for the negative class.
            py1 (torch.Tensor): Tensor representing the probability of the positive class.

        Returns:
            torch.Tensor: Expected value of the look-ahead uncertainty reduction for each query point.
        """
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

    def __init__(
        self,
        lb: Tensor,
        ub: Tensor,
        model: GPyTorchModel,
        lookahead_type: Literal["levelset", "posterior"] = "posterior",
        target: Optional[float] = None,
        query_set_size: Optional[int] = 256,
        Xq: Optional[torch.Tensor] = None,
        k: Optional[float] = 20.0,
    ) -> None:
        """
        model (GPyTorchModel): The gpytorch model to use.
        lookahead_type (Literal["levelset", "posterior"]): The type of look-ahead to perform (default is "posterior").
                - If the lookahead_type is "levelset", the acqf will consider the posterior probability that a point is above or below the target level set.
                - If the lookahead_type is "posterior", the acqf will consider the posterior probability that a point will be detected or not.
        target (float, optional): Threshold value to target in p-space. Default is None.
        query_set_size (int, optional): Number of points in the query set. Default is 256.
        Xq (torch.Tensor, optional): (m x d) global reference set. Default is None.
        k (float, optional): Scaling factor for the softmax approximation, controlling the "softness" of the maximum operation. Default is 20.0.
        """

        super().__init__(
            lb=lb,
            ub=ub,
            model=model,
            target=target,
            lookahead_type=lookahead_type,
            query_set_size=query_set_size,
            Xq=Xq,
        )
        self.k = k

    def _compute_acqf(
        self, Px: torch.Tensor, P1: torch.Tensor, P0: torch.Tensor, py1: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the SMOCU acquisition function.

        Args:
            Px (torch.Tensor): Tensor representing the current probability of class prediction.
            P1 (torch.Tensor): Tensor of look-ahead predictions for the positive class.
            P0 (torch.Tensor): Tensor of look-ahead predictions for the negative class.
            py1 (torch.Tensor): Tensor representing the probability of the positive class.

        Returns:
            torch.Tensor: Expected reduction in uncertainty, based on the softmax approximation of the maximum query.
        """
        current_softmax_query = (
            torch.logsumexp(self.k * torch.stack((Px, 1 - Px), dim=-1), dim=-1) / self.k
        )
        # expectation w.r.t. y* of the max of pq
        lookahead_pq1_softmax = (
            torch.logsumexp(self.k * torch.stack((P1, 1 - P1), dim=-1), dim=-1) / self.k
        )
        lookahead_pq0_softmax = (
            torch.logsumexp(self.k * torch.stack((P0, 1 - P0), dim=-1), dim=-1) / self.k
        )
        lookahead_softmax_query = (
            lookahead_pq1_softmax * py1 + lookahead_pq0_softmax * (1 - py1)
        )
        return (lookahead_softmax_query - current_softmax_query).mean(-1)


class BEMPS(GlobalLookaheadAcquisitionFunction):
    """
    BEMPS acquisition function given in:

        Tan, Wei, et al. "Diversity Enhanced Active Learning with Strictly Proper Scoring Rules."
        Advances in Neural Information Processing Systems 34 (2021).
    """

    def __init__(self, scorefun: Callable, *args, **kwargs) -> None:
        """
        scorefun (Callable): Scoring function to use for the BEMPS acquisition function.
        """
        super().__init__(*args, **kwargs)
        self.scorefun = scorefun

    def _compute_acqf(
        self, Px: torch.Tensor, P1: torch.Tensor, P0: torch.Tensor, py1: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the BEMPS acquisition function.

        Args:
            Px (torch.Tensor): Tensor representing the current probability of class prediction.
            P1 (torch.Tensor): Tensor of look-ahead predictions for the positive class.
            P0 (torch.Tensor): Tensor of look-ahead predictions for the negative class.
            py1 (torch.Tensor): Tensor representing the probability of the positive class.

        Returns:
            torch.Tensor: Expected improvement in the scoring function values, based on the current and look-ahead states.
        """
        current_score = self.scorefun(Px)
        lookahead_pq1_score = self.scorefun(P1)
        lookahead_pq0_score = self.scorefun(P0)
        lookahead_expected_score = lookahead_pq1_score * py1 + lookahead_pq0_score * (
            1 - py1
        )
        return (lookahead_expected_score - current_score).mean(-1)


@acqf_input_constructor(GlobalMI, GlobalSUR, ApproxGlobalSUR, EAVC, MOCU, SMOCU, BEMPS)
def construct_inputs_global_lookahead(
    model: GPyTorchModel,
    training_data: None,
    lookahead_type: Literal["levelset", "posterior"] = "levelset",
    target: Optional[float] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    query_set_size: Optional[int] = 256,
    Xq: Optional[torch.Tensor] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Constructs the input dictionary for initializing global lookahead acquisition functions.

    Args:
        model (GPyTorchModel): The gpytorch model to use.
        training_data (None): Placeholder for compatibility; not used in this function.
        lookahead_type (Literal["levelset", "posterior"]): The type of look-ahead to perform (default is "levelset").
                - If the lookahead_type is "levelset", the acqf will consider the posterior probability that a point is above or below the target level set.
                - If the lookahead_type is "posterior", the acqf will consider the posterior probability that a point will be detected or not.
        target (float, optional): Target threshold value in probability space. Default is None.
        posterior_transform (PosteriorTransform, optional): Optional transformation to apply to the posterior. Default is None.
        query_set_size (int, optional): Number of points in the query set. Default is 256.
        Xq (torch.Tensor, optional): (m x d) global reference set. If not provided, a Sobol sequence is generated. Default is None.

    Returns:
        Dict[str, Any]: Dictionary of constructed inputs for global lookahead acquisition functions.
    """

    lb = torch.tensor([bounds[0] for bounds in kwargs["bounds"]])
    ub = torch.tensor([bounds[1] for bounds in kwargs["bounds"]])
    if Xq is None and query_set_size is None:
        raise ValueError("Either Xq or query_set_size must be provided.")

    if Xq is None and query_set_size is not None:
        Xq = make_scaled_sobol(lb, ub, query_set_size)
    return {
        "model": model,
        "lookahead_type": lookahead_type,
        "target": target,
        "posterior_transform": posterior_transform,
        "query_set_size": query_set_size,
        "Xq": Xq,
    }
