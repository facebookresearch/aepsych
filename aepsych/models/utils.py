#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from aepsych.models.base import ModelProtocol
from botorch.models.approximate_gp import _select_inducing_points
from botorch.utils.sampling import draw_sobol_samples
from scipy.cluster.vq import kmeans2


def compute_p_quantile(
    f_mean: torch.Tensor, f_std: torch.Tensor, alpha: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Compute quantile of p in probit model

    For f ~ N(mu_f, sigma_f^2) and p = Phi(f), computes the alpha quantile of p
    using the formula

    x = Phi(mu_f + Phi^-1(alpha) * sigma_f),

    which solves for x such that P(p <= x) = alpha.

    A 95% CI for p can be computed as
    p_l = compute_p_quantile(f_mean, f_std, 0.025)
    p_u = compute_p_quantile(f_mean, f_std, 0.975)
    """
    norm = torch.distributions.Normal(0, 1)
    alpha = torch.tensor(alpha, dtype=f_mean.dtype)
    return norm.cdf(f_mean + norm.icdf(alpha) * f_std)


def select_inducing_points(
    inducing_size: int,
    model: Optional[ModelProtocol] = None,
    X: Optional[torch.Tensor] = None,
    bounds: Optional[Union[torch.Tensor, np.ndarray]] = None,
    method: str = "auto",
):
    with torch.no_grad():
        assert method in (
            "pivoted_chol",
            "kmeans++",
            "auto",
            "sobol",
        ), f"Inducing point method should be one of pivoted_chol, kmeans++, sobol, or auto; got {method}"

        if method == "sobol":
            assert bounds is not None, "Must pass bounds for sobol inducing points!"
            inducing_points = draw_sobol_samples(
                bounds=bounds, n=inducing_size, q=1
            ).squeeze()
            if len(inducing_points.shape) == 1:
                inducing_points = inducing_points.reshape(-1, 1)
            return inducing_points

        assert X is not None, "Must pass X for non-sobol inducing point selection!"
        # remove dupes from X, which is both wasteful for inducing points
        # and would break kmeans++
        unique_X = torch.unique(X, dim=0)
        if method == "auto":
            if unique_X.shape[0] <= inducing_size:
                return unique_X
            else:
                method = "kmeans++"

        if method == "pivoted_chol":
            assert model is not None and hasattr(
                model, "covar_module"
            ), "Must pass model with a covar_module for pivoted_chol inducing point selection!"
            inducing_points = _select_inducing_points(
                inputs=unique_X,
                covar_module=model.covar_module,
                num_inducing=inducing_size,
                input_batch_shape=torch.Size([]),
            )
        elif method == "kmeans++":
            # initialize using kmeans
            inducing_points = torch.tensor(
                kmeans2(unique_X.numpy(), inducing_size, minit="++")[0],
                dtype=X.dtype,
            )
        return inducing_points
