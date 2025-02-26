#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gpytorch
import torch
from aepsych.likelihoods import OrdinalLikelihood
from aepsych.models import GPClassificationModel


class OrdinalGPModel(GPClassificationModel):
    """
    Convenience wrapper for GPClassificationModel that hardcodes
    an ordinal likelihood, better priors for this setting, and
    adds a convenience method for computing outcome probabilities.

    TODO: at some point we should refactor posteriors so that things like
    OrdinalPosterior and MonotonicPosterior don't have to have their own
    model classes.
    """

    outcome_type = "ordinal"

    def __init__(self, likelihood=None, *args, **kwargs):
        """Initialize the OrdinalGPModel

        Args:
            likelihood (Likelihood): The likelihood function to use. If None defaults to
                Ordinal likelihood.
        """
        covar_module = kwargs.pop("covar_module", None)
        dim = kwargs.get("dim")
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
            *args,
            covar_module=covar_module,
            likelihood=likelihood,
            **kwargs,
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
