#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import gpytorch
import torch
from aepsych.config import ConfigurableMixin
from gpytorch.likelihoods import Likelihood
from torch.distributions import Categorical, Normal


class OrdinalLikelihood(Likelihood, ConfigurableMixin):
    """
    Ordinal likelihood, suitable for rating models (e.g. likert scales). Formally,
    .. math:: z_k(x\\mid f) := p(d_k < f(x) \\le d_{k+1}) = \\sigma(d_{k+1}-f(x)) - \\sigma(d_{k}-f(x)),
    where :math:`\\sigma()` is the link function (equivalent to the perceptual noise
    distribution in psychophysics terms), :math:`f(x)` is the latent GP evaluated at x,
    and :math:`d_k` is a learned cutpoint parameter for each level.
    """

    def __init__(self, n_levels: int, link: Callable | None = None) -> None:
        """Initialize OrdinalLikelihood.

        Args:
            n_levels (int): Number of levels in the ordinal scale.
            link (Callable, optional): Link function. Defaults to None.
        """
        super().__init__()
        self.n_levels = n_levels
        self.register_parameter(
            name="raw_cutpoint_deltas",
            parameter=torch.nn.Parameter(torch.abs(torch.randn(n_levels - 2))),
        )
        self.register_constraint("raw_cutpoint_deltas", gpytorch.constraints.Positive())
        self.link = link or Normal(0, 1).cdf

    @property
    def cutpoints(self) -> torch.Tensor:
        cutpoint_deltas = self.raw_cutpoint_deltas_constraint.transform(
            self.raw_cutpoint_deltas
        )
        # for identification, the first cutpoint is 0
        return torch.cat(
            (torch.tensor([0]).to(cutpoint_deltas), torch.cumsum(cutpoint_deltas, 0))
        )

    def forward(self, function_samples: torch.Tensor, *params, **kwargs) -> Categorical:
        """Forward pass for Ordinal

        Args:
            function_samples (torch.Tensor): Function samples.

        Returns:
            Categorical: Categorical distribution object.
        """

        # this whole thing can probably be some clever batched thing, meh
        probs = torch.zeros(*function_samples.size(), self.n_levels).to(
            function_samples
        )

        probs[..., 0] = self.link(self.cutpoints[0] - function_samples)

        for i in range(1, self.n_levels - 1):
            probs[..., i] = self.link(self.cutpoints[i] - function_samples) - self.link(
                self.cutpoints[i - 1] - function_samples
            )
        probs[..., -1] = 1 - self.link(self.cutpoints[-1] - function_samples)
        res = Categorical(probs=probs)
        return res
