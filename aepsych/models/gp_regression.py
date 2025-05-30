#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from copy import deepcopy
from typing import Any

import gpytorch
import torch
from aepsych.factory import DefaultMeanCovarFactory
from aepsych.models.base import AEPsychModelMixin
from aepsych.utils_logging import getLogger
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.models import ExactGP

logger = getLogger()


class GPRegressionModel(AEPsychModelMixin, ExactGP):
    """GP Regression model for continuous outcomes, using exact inference."""

    _num_outputs = 1
    stimuli_per_trial = 1
    outcome_type = "continuous"

    def __init__(
        self,
        dim: int,
        mean_module: gpytorch.means.Mean | None = None,
        covar_module: gpytorch.kernels.Kernel | None = None,
        likelihood: Likelihood | None = None,
        max_fit_time: float | None = None,
        optimizer_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the GP regression model

        Args:
            dim (int): The number of dimensions in the parameter space.
            mean_module (gpytorch.means.Mean, optional): GP mean class. Defaults to a constant with a normal prior.
            covar_module (gpytorch.kernels.Kernel, optional): GP covariance kernel class. Defaults to scaled RBF with a
                gamma prior.
            likelihood (Likelihood, optional): The likelihood function to use. If None defaults to
                Gaussian likelihood.
            max_fit_time (float, optional): The maximum amount of time, in seconds, to spend fitting the model. If None,
                there is no limit to the fitting time.
            optimizer_options (dict[str, Any], optional): Optimizer options to pass to the SciPy optimizer during
                fitting. Assumes we are using L-BFGS-B.
        """
        self.dim = dim

        if likelihood is None:
            likelihood = GaussianLikelihood()

        super().__init__(None, None, likelihood)

        self.max_fit_time = max_fit_time

        self.optimizer_options = (
            {"options": optimizer_options} if optimizer_options else {"options": {}}
        )

        if mean_module is None or covar_module is None:
            factory = DefaultMeanCovarFactory(
                dim=self.dim,
                stimuli_per_trial=self.stimuli_per_trial,
            )
            default_mean = factory.get_mean()
            default_covar = factory.get_covar()

        self.likelihood = likelihood
        self.mean_module = mean_module or default_mean
        self.covar_module = covar_module or default_covar

        self._fresh_state_dict = deepcopy(self.state_dict())
        self._fresh_likelihood_dict = deepcopy(self.likelihood.state_dict())

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs) -> None:
        """Fit underlying model.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.LongTensor): Responses.
        """
        self.set_train_data(train_x, train_y)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        return self._fit_mll(mll, self.optimizer_options, **kwargs)
