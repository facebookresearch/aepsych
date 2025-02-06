#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import gpytorch
import torch
from aepsych.config import Config
from aepsych.factory.default import default_mean_covar_factory
from aepsych.models.base import AEPsychModelMixin
from aepsych.models.inducing_points import GreedyVarianceReduction
from aepsych.models.inducing_points.base import InducingPointAllocator
from aepsych.utils_logging import getLogger
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

logger = getLogger()


class VariationalGPModel(AEPsychModelMixin, ApproximateGP):
    """Base GP model with variational inference"""

    _batch_size = 1

    def __init__(
        self,
        dim: int,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        inducing_point_method: Optional[InducingPointAllocator] = None,
        inducing_size: int = 100,
        max_fit_time: Optional[float] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the GP Classification model

        Args:
            dim (int): The number of dimensions in the parameter space.
            mean_module (gpytorch.means.Mean, optional): GP mean class. Defaults to a constant with a normal prior.
            covar_module (gpytorch.kernels.Kernel, optional): GP covariance kernel class. Defaults to scaled RBF with a
                gamma prior.
            likelihood (gpytorch.likelihood.Likelihood, optional): The likelihood function to use. If None defaults to
                Gaussian likelihood.
            inducing_point_method (InducingPointAllocator, optional): The method to use for selecting inducing points.
                If not set, a GreedyVarianceReduction is made.
            inducing_size (int): Number of inducing points. Defaults to 100.
            max_fit_time (float, optional): The maximum amount of time, in seconds, to spend fitting the model. If None,
                there is no limit to the fitting time.
            optimizer_options (Dict[str, Any], optional): Optimizer options to pass to the SciPy optimizer during
                fitting. Assumes we are using L-BFGS-B.
        """
        self.dim = dim
        self.max_fit_time = max_fit_time
        self.inducing_size = inducing_size

        self.optimizer_options = (
            {"options": optimizer_options} if optimizer_options else {"options": {}}
        )

        if likelihood is None:
            likelihood = GaussianLikelihood()

        if mean_module is None or covar_module is None:
            default_mean, default_covar = default_mean_covar_factory(
                dim=self.dim, stimuli_per_trial=self.stimuli_per_trial
            )

        self.inducing_point_method = inducing_point_method or GreedyVarianceReduction(
            dim=self.dim
        )
        inducing_points = self.inducing_point_method.allocate_inducing_points(
            num_inducing=self.inducing_size,
            covar_module=covar_module or default_covar,
        )

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=torch.Size([self._batch_size])
        ).to(inducing_points)

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        )
        super().__init__(variational_strategy)

        self.likelihood = likelihood
        self.mean_module = mean_module or default_mean
        self.covar_module = covar_module or default_covar

        self._fresh_state_dict = deepcopy(self.state_dict())
        self._fresh_likelihood_dict = deepcopy(self.likelihood.state_dict())

    def _reset_hyperparameters(self) -> None:
        """Reset hyperparameters to their initial values."""
        # warmstart_hyperparams affects hyperparams but not the variational strat,
        # so we keep the old variational strat (which is only refreshed
        # if warmstart_induc=False).
        vsd = self.variational_strategy.state_dict()  # type: ignore
        vsd_hack = {f"variational_strategy.{k}": v for k, v in vsd.items()}
        state_dict = deepcopy(self._fresh_state_dict)
        state_dict.update(vsd_hack)
        self.load_state_dict(state_dict)
        self.likelihood.load_state_dict(self._fresh_likelihood_dict)

    def _reset_variational_strategy(self) -> None:
        if self.train_inputs is not None:
            # remember original device
            device = self.device
            inducing_points = self.inducing_point_method.allocate_inducing_points(
                num_inducing=self.inducing_size,
                covar_module=self.covar_module,
                inputs=self.train_inputs[0],
            ).to(device)
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(0), batch_shape=torch.Size([self._batch_size])
            ).to(device)
            self.variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ).to(device)

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        warmstart_hyperparams: bool = False,
        warmstart_induc: bool = False,
        **kwargs,
    ) -> None:
        """Fit underlying model.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.LongTensor): Responses.
            warmstart_hyperparams (bool): Whether to reuse the previous hyperparameters (True) or fit from scratch
                (False). Defaults to False.
            warmstart_induc (bool): Whether to reuse the previous inducing points or fit from scratch (False).
                Defaults to False.
        """
        self.set_train_data(train_x, train_y)

        # by default we reuse the model state and likelihood. If we
        # want a fresh fit (no warm start), copy the state from class initialization.
        if not warmstart_hyperparams:
            self._reset_hyperparameters()

        if not warmstart_induc or (
            self.inducing_point_method.last_allocator_used is None
        ):
            self._reset_variational_strategy()

        n = train_y.shape[0]
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, n)

        if "optimizer_kwargs" in kwargs:
            self._fit_mll(mll, **kwargs)
        else:
            self._fit_mll(mll, optimizer_kwargs=self.optimizer_options, **kwargs)

    def update(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs):
        """Perform a warm-start update of the model from previous fit.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.Tensor): Responses.
        """
        return self.fit(
            train_x, train_y, warmstart_hyperparams=True, warmstart_induc=True, **kwargs
        )
