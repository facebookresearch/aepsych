#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from gpytorch.models import GP
from torch import Tensor

from .bvn import bvn_cdf


def posterior_at_xstar_xq(
    model: GP,
    Xstar: Tensor,
    Xq: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Evaluate the posteriors of f at single point Xstar and set of points Xq.

    Args:
        model: The model to evaluate.
        Xstar: (b x 1 x d) tensor.
        Xq: (b x m x d) tensor.

    Returns:
        Mu_s: (b x 1) mean at Xstar.
        Sigma2_s: (b x 1) variance at Xstar.
        Mu_q: (b x m) mean at Xq.
        Sigma2_q: (b x m) variance at Xq.
        Sigma_sq: (b x m) covariance between Xstar and each point in Xq.
    """
    # Evaluate posterior and extract needed components
    Xext = torch.cat((Xstar, Xq), dim=-2)
    posterior = model.posterior(Xext)
    mu = posterior.mean[..., :, 0]
    Mu_s = mu[..., 0].unsqueeze(-1)
    Mu_q = mu[..., 1:]
    Cov = posterior.mvn.covariance_matrix
    Sigma2_s = Cov[..., 0, 0].unsqueeze(-1)
    Sigma2_q = torch.diagonal(Cov[..., 1:, 1:], dim1=-1, dim2=-2)
    Sigma_sq = Cov[..., 0, 1:]
    return Mu_s, Sigma2_s, Mu_q, Sigma2_q, Sigma_sq


def lookahead_at_xstar(
    model: GP,
    Xstar: Tensor,
    Xq: Tensor,
    gamma: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Evaluate the look-ahead level-set posterior at Xq given observation at xstar.

    Args:
        model: The model to evaluate.
        Xstar: (b x 1 x d) observation point.
        Xq: (b x m x d) reference points.
        gamma: Threshold in f-space.

    Returns:
        Px: (b x m) Level-set posterior at Xq, before observation at xstar.
        P1: (b x m) Level-set posterior at Xq, given observation of 1 at xstar.
        P0: (b x m) Level-set posterior at Xq, given observation of 0 at xstar.
        py1: (b x 1) Probability of observing 1 at xstar.
    """
    Mu_s, Sigma2_s, Mu_q, Sigma2_q, Sigma_sq = posterior_at_xstar_xq(model, Xstar, Xq)

    # Compute look-ahead components
    Norm = torch.distributions.Normal(0, 1)
    Sigma_q = torch.sqrt(Sigma2_q)
    b_q = (gamma - Mu_q) / Sigma_q
    Phi_bq = Norm.cdf(b_q)
    denom = torch.sqrt(1 + Sigma2_s)
    a_s = Mu_s / denom
    Phi_as = Norm.cdf(a_s)
    Z_rho = -Sigma_sq / (Sigma_q * denom)
    Z_qs = bvn_cdf(a_s, b_q, Z_rho)

    Px = Phi_bq
    py1 = Phi_as
    P1 = Z_qs / py1
    P0 = (Phi_bq - Z_qs) / (1 - py1)
    return Px, P1, P0, py1


def approximate_lookahead_at_xstar(
    model: GP,
    Xstar: Tensor,
    Xq: Tensor,
    gamma: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    The look-ahead posterior approximation of Lyu et al.

    Args:
        model: The model to evaluate.
        Xstar: (b x 1 x d) observation point.
        Xq: (b x m x d) reference points.
        gamma: Threshold in f-space.

    Returns:
        Px: (b x m) Level-set posterior at Xq, before observation at xstar.
        P1: (b x m) Level-set posterior at Xq, given observation of 1 at xstar.
        P0: (b x m) Level-set posterior at Xq, given observation of 0 at xstar.
        py1: (b x 1) Probability of observing 1 at xstar.
    """
    Mu_s, Sigma2_s, Mu_q, Sigma2_q, Sigma_sq = posterior_at_xstar_xq(model, Xstar, Xq)

    Norm = torch.distributions.Normal(0, 1)
    Mu_s_pdf = torch.exp(Norm.log_prob(Mu_s))
    Mu_s_cdf = Norm.cdf(Mu_s)

    # Formulae from the supplement of the paper (Result 2)
    vnp1_p = Mu_s_pdf ** 2 / Mu_s_cdf ** 2 + Mu_s * Mu_s_pdf / Mu_s_cdf  # (C.4)
    p_p = Norm.cdf(Mu_s / torch.sqrt(1 + Sigma2_s))  # (C.5)

    vnp1_n = Mu_s_pdf ** 2 / (1 - Mu_s_cdf) ** 2 - Mu_s * Mu_s_pdf / (
        1 - Mu_s_cdf
    )  # (C.6)
    p_n = 1 - p_p  # (C.7)

    vtild = vnp1_p * p_p + vnp1_n * p_n

    Sigma2_q_np1 = Sigma2_q - Sigma_sq ** 2 / ((1 / vtild) + Sigma2_s)  # (C.8)

    Px = Norm.cdf((gamma - Mu_q) / torch.sqrt(Sigma2_q))
    P1 = Norm.cdf((gamma - Mu_q) / torch.sqrt(Sigma2_q_np1))
    P0 = P1  # Same because we ignore value of y in this approximation
    py1 = 0.5 * torch.ones(*Px.shape[:-1], 1)  # Value doesn't matter because P1 = P0
    return Px, P1, P0, py1
