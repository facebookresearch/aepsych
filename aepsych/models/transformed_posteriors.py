#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from botorch.posteriors import Posterior, TransformedPosterior
from torch import Tensor
from torch.distributions import Normal

from .utils import bernoulli_probit_prob_transform


class BernoulliProbitProbabilityPosterior(TransformedPosterior):
    """The posterior of a Bernoulli-probit response model in probability space."""

    def _bernoulli_probit_prob_mean(self, mean, var):
        return bernoulli_probit_prob_transform(mean, var)[0]

    def _bernoulli_probit_prob_var(self, mean, var):
        return bernoulli_probit_prob_transform(mean, var)[1]

    def __init__(self, posterior):
        """Constructor for the transformed posterior.

        Args:
            posterior (Posterior): The latent posterior of a Bernoulli-probit response model.
        """
        super().__init__(
            posterior=posterior,
            sample_transform=Normal(0, 1).cdf,
            mean_transform=self._bernoulli_probit_prob_mean,
            variance_transform=self._bernoulli_probit_prob_var,
        )


class MCTransformedPosterior(TransformedPosterior):
    """A posterior, with mean and variance transformed by applying some transform to Monte Carlo samples, and then taking the mean and variance of the transformed values."""

    def _mc_mean(self, mean, var):
        samps = self.sample(torch.Size([self.num_samples]))
        return samps.mean(0)

    def _mc_var(self, mean, var):
        samps = self.sample(torch.Size([self.num_samples]))
        return samps.var(0)

    def __init__(
        self,
        posterior: Posterior,
        sample_transform: Callable[[Tensor], Tensor] | None = None,
        num_samples: int = 1000,
    ):
        """Constructor for the transformed posterior.

        Args:
            posterior (Posterior): The latent posterior to be transformed.
            sample_transform (Callable[[Tensor], Tensor], optional): A callable applying a sample-level transform to a
                `sample_shape x batch_shape x q x m`-dim tensor of samples from
                the original posterior, returning a tensor of samples of the
                same shape. If None, defaults to the identity function
            num_samples: The number of samples to estimate the mean and variance from.
        """
        # Default to identity function
        if sample_transform is None:

            def sample_transform(x):
                return x

        super().__init__(
            posterior=posterior,
            sample_transform=sample_transform,
            mean_transform=self._mc_mean,
            variance_transform=self._mc_var,
        )
        self.num_samples = num_samples
