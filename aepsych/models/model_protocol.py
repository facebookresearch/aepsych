#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Protocol

import torch
from botorch.posteriors import Posterior
from gpytorch.likelihoods import Likelihood

from .transformed_posteriors import TransformedPosterior


class ModelProtocol(Protocol):
    @property
    def _num_outputs(self) -> int:
        pass

    @property
    def outcome_type(self) -> str:
        pass

    @property
    def extremum_solver(self) -> str:
        pass

    @property
    def train_inputs(self) -> torch.Tensor:
        pass

    @property
    def dim(self) -> int:
        pass

    @property
    def device(self) -> torch.device:
        pass

    def posterior(self, X: torch.Tensor) -> Posterior:
        pass

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def predict_probability(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def predict_transform(
        self,
        x: torch.Tensor,
        **kwargs,
    ):
        pass

    @property
    def stimuli_per_trial(self) -> int:
        pass

    @property
    def likelihood(self) -> Likelihood:
        pass

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        pass

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs: Any) -> None:
        pass

    def update(
        self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs: Any
    ) -> None:
        pass
