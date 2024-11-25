#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math import pi as _pi
from typing import Tuple

import torch

inv_2pi = 1 / (2 * _pi)
_neg_inv_sqrt2 = -1 / (2**0.5)


def _gauss_legendre20(dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the abscissae and weights for the Gauss-Legendre quadrature of order 20.

    Args:
        dtype (torch.dtype): The desired data type of the output tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - abscissae: The quadrature points.
            - weights: The corresponding weights for each quadrature point.
    """
    _abscissae = torch.tensor(
        [
            0.9931285991850949,
            0.9639719272779138,
            0.9122344282513259,
            0.8391169718222188,
            0.7463319064601508,
            0.6360536807265150,
            0.5108670019508271,
            0.3737060887154196,
            0.2277858511416451,
            0.07652652113349733,
        ],
        dtype=dtype,
    )

    _weights = torch.tensor(
        [
            0.01761400713915212,
            0.04060142980038694,
            0.06267204833410906,
            0.08327674157670475,
            0.1019301198172404,
            0.1181945319615184,
            0.1316886384491766,
            0.1420961093183821,
            0.1491729864726037,
            0.1527533871307259,
        ],
        dtype=dtype,
    )
    abscissae = torch.cat([1.0 - _abscissae, 1.0 + _abscissae], dim=0)
    weights = torch.cat([_weights, _weights], dim=0)
    return abscissae, weights


def _ndtr(x: torch.Tensor) -> torch.Tensor:
    """
    Standard normal CDF. Called <phid> in Genz's original code.

    Args:
        x (torch.Tensor): Input tensor of values.

    Returns:
        torch.Tensor: CDF values for each element in the input tensor.
    """
    return 0.5 * torch.erfc(_neg_inv_sqrt2 * x)


def _bvnu(
    dh: torch.Tensor,
    dk: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """
    Primary subroutine for bvnu()

    Args:
        dh (torch.Tensor): Input tensor representing the first variable values.
        dk (torch.Tensor): Input tensor representing the second variable values.
        r (torch.Tensor): Input tensor for the correlation coefficient.

    Returns:
        torch.Tensor: Approximated bivariate normal CDF values.
    """
    # Precompute some terms
    h = dh
    k = dk
    hk = h * k

    x, w = _gauss_legendre20(dtype=dh.dtype)
    x, w = x.to(dh), w.to(dh)

    asr = 0.5 * torch.asin(r)
    sn = torch.sin(asr[..., None] * x)
    res = (sn * hk[..., None] - 0.5 * (h**2 + k**2)[..., None]) / (1 - sn**2)
    res = torch.sum(w * torch.exp(res), dim=-1)
    res = res * inv_2pi * asr + _ndtr(-h) * _ndtr(-k)

    return torch.clip(res, 0, 1)


def bvn_cdf(
    xu: torch.Tensor,
    yu: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate the bivariate normal CDF.

    WARNING: Implements only the routine for moderate levels of correlation. Will be
    inaccurate and should not be used for correlations larger than 0.925.

    Standard (mean 0, var 1) bivariate normal distribution with correlation r.
    Evaluated from -inf to xu, and -inf to yu.

    Based on function developed by Alan Genz:
    http://www.math.wsu.edu/faculty/genz/software/matlab/bvn.m

    based in turn on
    Drezner, Z and G.O. Wesolowsky, (1989),
    On the computation of the bivariate normal inegral,
    Journal of Statist. Comput. Simul. 35, pp. 101-107.

    Args:
        xu (torch.Tensor): Upper limits for cdf evaluation in x
        yu (torch.Tensor): Upper limits for cdf evaluation in y
        r (torch.Tensor): BVN correlation

    Returns:
        Tensor of cdf evaluations of same size as xu, yu, and r.
    """
    p = 1 - _ndtr(-xu) - _ndtr(-yu) + _bvnu(xu, yu, r)
    return torch.clip(p, 0, 1)
