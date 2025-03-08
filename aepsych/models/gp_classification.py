#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gpytorch
import torch
from aepsych.models.inducing_points.base import InducingPointAllocator
from aepsych.utils_logging import getLogger
from botorch.posteriors import TransformedPosterior
from gpytorch.likelihoods import BernoulliLikelihood, Likelihood

from .transformed_posteriors import BernoulliProbitProbabilityPosterior
from .variationalgp import VariationalGPModel

logger = getLogger()


class GPClassificationModel(VariationalGPModel):
    """Probit-GP model with variational inference.

    From a conventional ML perspective this is a GP Classification model,
    though in the psychophysics context it can also be thought of as a
    nonlinear generalization of the standard linear model for 1AFC or
    yes/no trials.

    For more on variational inference, see e.g.
    https://docs.gpytorch.ai/en/v1.1.1/examples/04_Variational_and_Approximate_GPs/
    """

    _num_outputs = 1
    stimuli_per_trial = 1
    outcome_type = "binary"

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
                Bernouli likelihood. This should not be modified unless you know what you're doing.
            inducing_point_method (InducingPointAllocator, optional): The method to use for selecting inducing points.
                If not set, a GreedyVarianceReduction is made.
            inducing_size (int): Number of inducing points. Defaults to 100.
            max_fit_time (float, optional): The maximum amount of time, in seconds, to spend fitting the model. If None,
                there is no limit to the fitting time.
            optimizer_options (Dict[str, Any], optional): Optimizer options to pass to the SciPy optimizer during
                fitting. Assumes we are using L-BFGS-B.
        """
        if likelihood is None:
            likelihood = BernoulliLikelihood()

        super().__init__(
            dim=dim,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            inducing_point_method=inducing_point_method,
            inducing_size=inducing_size,
            max_fit_time=max_fit_time,
            optimizer_options=optimizer_options,
        )

    def predict(
        self, x: torch.Tensor, probability_space: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            probability_space (bool): Return outputs in units of
                response probability instead of latent function value. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at query points.
        """

        if not probability_space:
            return super().predict(x)

        return self.predict_transform(
            x=x, transformed_posterior_cls=BernoulliProbitProbabilityPosterior
        )

    def predict_transform(
        self,
        x: torch.Tensor,
        transformed_posterior_cls: Optional[
            type[TransformedPosterior]
        ] = BernoulliProbitProbabilityPosterior,
        **transform_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance under some tranformation.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            transformed_posterior_cls (TransformedPosterior type, optional): The type of transformation to apply to the posterior.
                Note that you should give TransformedPosterior itself, rather than an instance. Defaults to None, in which case no
                transformation is applied.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed posterior mean and variance at query points.
        """

        return super().predict_transform(
            x=x, transformed_posterior_cls=transformed_posterior_cls
        )

    def predict_probability(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance in probability space.

        Args:
            x (torch.Tensor): Points at which to predict from the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at query points.
        """
        return self.predict(x, probability_space=True)
