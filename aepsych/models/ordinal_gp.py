#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import gpytorch
import torch
from aepsych.likelihoods import OrdinalLikelihood
from aepsych.models.inducing_points.base import InducingPointAllocator
from aepsych.models.variational_gp import VariationalGPModel
from gpytorch.likelihoods import Likelihood


class OrdinalGPModel(VariationalGPModel):
    """
    A convenience wrapper about the base VariationalGPs with default covariance module
    and likelihood to fit a Ordinal model.
    """

    outcome_type = "ordinal"
    _num_outputs = 1
    _stimuli_per_trial = 1

    def __init__(
        self,
        dim: int,
        mean_module: gpytorch.means.Mean | None = None,
        covar_module: gpytorch.kernels.Kernel | None = None,
        likelihood: Likelihood | None = None,
        mll_class: gpytorch.mlls.MarginalLogLikelihood | None = None,
        inducing_point_method: InducingPointAllocator | None = None,
        inducing_size: int = 100,
        max_fit_time: float | None = None,
        optimizer_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the GP Classification model

        Args:
            dim (int): The number of dimensions in the parameter space.
            mean_module (gpytorch.means.Mean, optional): GP mean class. Defaults to a constant with a normal prior.
            covar_module (gpytorch.kernels.Kernel, optional): GP covariance kernel class. Defaults to scaled RBF with a
                gamma prior.
            likelihood (gpytorch.likelihood.Likelihood, optional): The likelihood function to use. If None defaults to
                Gaussian likelihood.
            mll_class (gpytorch.mlls.MarginalLogLikelihood, optional): The approximate marginal log likelihood class to
                use. If None defaults to VariationalELBO.
            inducing_point_method (InducingPointAllocator, optional): The method to use for selecting inducing points.
                If not set, a GreedyVarianceReduction is made.
            inducing_size (int): Number of inducing points. Defaults to 100.
            max_fit_time (float, optional): The maximum amount of time, in seconds, to spend fitting the model. If None,
                there is no limit to the fitting time.
            optimizer_options (dict[str, Any], optional): Optimizer options to pass to the SciPy optimizer during
                fitting. Assumes we are using L-BFGS-B.
        """
        if covar_module is None:
            ls_prior = gpytorch.priors.GammaPrior(concentration=1.5, rate=3.0)
            ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate
            ls_constraint = gpytorch.constraints.Positive(
                transform=None, initial_value=ls_prior_mode
            )

            # no outputscale due to shift identifiability in d.
            covar_module = gpytorch.kernels.RBFKernel(
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint,
                ard_num_dims=dim,
            )

        if likelihood is None:
            likelihood = OrdinalLikelihood(n_levels=5)

        super().__init__(
            dim=dim,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            mll_class=mll_class,
            inducing_point_method=inducing_point_method,
            inducing_size=inducing_size,
            max_fit_time=max_fit_time,
            optimizer_options=optimizer_options,
        )

    def predict_probs(self, xgrid: torch.Tensor) -> torch.Tensor:
        """Predict probabilities of each ordinal level at xgrid

        Args:
            xgrid (torch.Tensor): Tensor of input points to predict at

        Returns:
            torch.Tensor: Tensor of probabilities of each ordinal level at xgrid
        """
        fmean, fvar = self.predict(xgrid)
        return self.calculate_probs(fmean, fvar)

    def calculate_probs(self, fmean: torch.Tensor, fvar: torch.Tensor) -> torch.Tensor:
        """Calculate probabilities of each ordinal level given a mean and variance

        Args:
            fmean (torch.Tensor): Mean of the latent function
            fvar (torch.Tensor): Variance of the latent function

        Returns:
            torch.Tensor: Tensor of probabilities of each ordinal level
        """
        fsd = torch.sqrt(1 + fvar)
        probs = torch.zeros(*fmean.size(), self.likelihood.n_levels)

        probs[..., 0] = self.likelihood.link(
            (self.likelihood.cutpoints[0] - fmean) / fsd
        )

        for i in range(1, self.likelihood.n_levels - 1):
            probs[..., i] = self.likelihood.link(
                (self.likelihood.cutpoints[i] - fmean) / fsd
            ) - self.likelihood.link((self.likelihood.cutpoints[i - 1] - fmean) / fsd)
        probs[..., -1] = 1 - self.likelihood.link(
            (self.likelihood.cutpoints[-1] - fmean) / fsd
        )
        return probs
