#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from typing import Any

import gpytorch
import torch
from aepsych.config import Config
from aepsych.likelihoods.mixed import ListLikelihood, MixedVariationalELBO
from aepsych.models.inducing_points import FixedPlusAllocator, GreedyVarianceReduction
from aepsych.models.inducing_points.base import InducingPointAllocator
from aepsych.utils_logging import getLogger
from botorch.posteriors import TransformedPosterior
from gpytorch.likelihoods import (
    BernoulliLikelihood,
    FixedNoiseGaussianLikelihood,
    Likelihood,
)

from .transformed_posteriors import BernoulliProbitProbabilityPosterior
from .variational_gp import VariationalGPModel

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
        mean_module: gpytorch.means.Mean | None = None,
        covar_module: gpytorch.kernels.Kernel | None = None,
        likelihood: Likelihood | None = None,
        mll_class: gpytorch.mlls.MarginalLogLikelihood | None = None,
        inducing_point_method: InducingPointAllocator | None = None,
        inducing_size: int = 100,
        constraint_locations: torch.Tensor | None = None,
        constraint_values: torch.Tensor | None = None,
        constraint_strengths: torch.Tensor | None = None,
        max_fit_time: float | None = None,
        optimizer_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the GP Classification model

        Args:
            dim (int): The number of dimensions in the parameter space.
            mean_module (gpytorch.means.Mean, optional): GP mean class. Defaults to a constant with a normal prior.
            covar_module (gpytorch.kernels.Kernel, optional): GP covariance kernel class. Defaults to scaled RBF with a
                gamma prior.
            likelihood (Likelihood, optional): The likelihood function to use. If None defaults to
                Bernouli likelihood. This should not be modified unless you know what you're doing.
            mll_class (gpytorch.mlls.MarginalLogLikelihood, optional): The approximate marginal log likelihood class to
                use. If None defaults to VariationalELBO.
            inducing_point_method (InducingPointAllocator, optional): The method to use for selecting inducing points.
                If not set, a GreedyVarianceReduction is made.
            inducing_size (int): Number of inducing points. Defaults to 100.
            constraint_locations (torch.Tensor, optional): Locations at which to constrain the latent function. Defaults
                to None.
            constraint_values (torch.Tensor, optional): Values at which to constrain the latent function. There must be
                one value for each constraint location. Assumed to be in probability space unless likelihood is not None,
                If likelihood is set, no transformations to these values will be done. Defaults to None.
            constraint_strengths (torch.Tensor, optional): Strength of the constraints. There must be one value for each
                constraint location. If set to None, constraint strength will be based on the value (0.2 * value + 0.1).
            max_fit_time (float, optional): The maximum amount of time, in seconds, to spend fitting the model. If None,
                there is no limit to the fitting time.
            optimizer_options (dict[str, Any], optional): Optimizer options to pass to the SciPy optimizer during
                fitting. Assumes we are using L-BFGS-B.
        """
        self.constraint_locations = constraint_locations
        self.constraint_values = constraint_values
        self.constraint_strengths = constraint_strengths

        if likelihood is None:
            likelihood = BernoulliLikelihood()

            if self.constraint_values is not None:
                # Values are clamped to -5 to 5 to handle 0 and 1 not becoming inf
                self.constraint_values = torch.clamp(
                    torch.distributions.Normal(0, 1).icdf(self.constraint_values), -5, 5
                )

        if self.constraint_locations is not None and mll_class is not None:
            raise ValueError(
                "Cannot set both constraints and mll_class. "
                "To use constraints, we create a mixed variational ELBO for the MLL."
            )

        # Check if constraints are all the same size
        if self.constraint_locations is not None and self.constraint_values is not None:
            # Constraints need to be 2D
            if self.constraint_locations.dim() != 2:
                self.constraint_locations = self.constraint_locations.unsqueeze(0)

            if self.constraint_values.dim() != 2:
                self.constraint_values = self.constraint_values.unsqueeze(1)

            # Needs the same n
            if self.constraint_locations.shape[0] != self.constraint_values.shape[0]:
                raise ValueError("Constraint locations and values must have the same n")

            # Manual strengths need to match values
            if self.constraint_strengths is not None:
                if (
                    self.constraint_values.shape[0]
                    != self.constraint_strengths.shape[0]
                ):
                    raise ValueError(
                        "Constraint locations, values, and strengths must have the same n"
                    )
            else:
                # Make strengths based on values
                self.constraint_strengths = (0.2 * self.constraint_values + 0.1) ** 2

            # Constraint strengths need to be exactly 1D
            if self.constraint_strengths.dim() == 2:
                self.constraint_strengths = self.constraint_strengths.squeeze(1)

            # Replace necessary objects
            mll_class = MixedVariationalELBO
            likelihood = ListLikelihood(
                likelihoods=[
                    likelihood,
                    FixedNoiseGaussianLikelihood(noise=self.constraint_strengths),
                ],
            )
            if inducing_point_method is None:
                inducing_point_method = FixedPlusAllocator(
                    dim=dim,
                    points=self.constraint_locations,
                    main_allocator=GreedyVarianceReduction,
                )
            else:
                inducing_point_method = FixedPlusAllocator(
                    dim=dim,
                    points=self.constraint_locations,
                    main_allocator=inducing_point_method,
                )

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

    def set_train_data(
        self,
        inputs: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        strict: bool = True,
    ) -> None:
        """Set the training data for the model.

        Args:
            inputs (torch.Tensor, optional): The new training inputs X.
            targets (torch.Tensor, optional): The new training targets Y.
            strict (bool): Whether to strictly enforce the device of the inputs and targets.
        """
        if not strict:
            warnings.warn(
                "strict set to false, but set_train_data will always follow device",
                stacklevel=2,
            )

        if self.constraint_locations is not None and self.constraint_values is not None:
            if inputs is not None:
                inputs = torch.cat((inputs, self.constraint_locations), dim=0)
            if targets is not None:
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                targets = torch.cat((targets, self.constraint_values), dim=0)

        if inputs is None:
            self.train_inputs = None
        else:
            self.train_inputs = (inputs,)
        self.train_targets = targets

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

        if self.constraint_locations is not None and self.constraint_values is not None:
            mll = MixedVariationalELBO(
                likelihood=self.likelihood,
                task_indices=[
                    torch.arange(n),
                    torch.arange(n, n + self.constraint_values.shape[0]),
                ],
                model=self,
                num_data=n,
            )
        else:
            mll = self.mll_class(self.likelihood, self, n)

        if "optimizer_kwargs" in kwargs:
            self._fit_mll(mll, **kwargs)
        else:
            self._fit_mll(mll, optimizer_kwargs=self.optimizer_options, **kwargs)

    def predict(
        self, x: torch.Tensor, probability_space: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            probability_space (bool): Return outputs in units of
                response probability instead of latent function value. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at query points.
        """

        if not probability_space:
            return super().predict(x)

        return self.predict_transform(
            x=x, transformed_posterior_cls=BernoulliProbitProbabilityPosterior
        )

    def predict_transform(
        self,
        x: torch.Tensor,
        transformed_posterior_cls: type[TransformedPosterior]
        | None = BernoulliProbitProbabilityPosterior,
        **transform_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance under some tranformation.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            transformed_posterior_cls (type[TransformedPosterior], optional): The type of transformation to apply to the posterior.
                Note that you should give TransformedPosterior itself, rather than an instance. Defaults to BernoulliProbitProbabilityPosterior.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Transformed posterior mean and variance at query points.
        """

        return super().predict_transform(
            x=x, transformed_posterior_cls=transformed_posterior_cls
        )

    def predict_probability(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance in probability space.

        Args:
            x (torch.Tensor): Points at which to predict from the model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at query points.
        """
        return self.predict(x, probability_space=True)

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get configuration options for the model.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): The name of the strategy to warm start (Not actually optional here.)
            options (dict[str, Any], optional): options are ignored.

        Raises:
            ValueError: the name of the strategy is necessary to identify warm start search criteria.
            KeyError: the config specified this strategy should be warm started but the associated config section wasn't defined.

        Returns:
            dict[str, Any]: a dictionary of the search criteria described in the experiment's config
        """
        name = name or cls.__name__
        options = super().get_config_options(config, name, options)

        constraint_factory = config.getobj(name, "constraint_factory", fallback=None)
        if constraint_factory is not None:
            factory_locations, factory_values, factory_strengths = constraint_factory(
                config
            )

            # If constraint values already exist, simply combine them with the factory values
            if (
                "constraint_locations" in options
                and options["constraint_locations"] is not None
            ):
                # Ensure existing locations are 2D
                if options["constraint_locations"].dim() != 2:
                    options["constraint_locations"] = options[
                        "constraint_locations"
                    ].unsqueeze(0)

                # Ensure values and strengths are 2D
                if options["constraint_values"].dim() != 2:
                    options["constraint_values"] = options[
                        "constraint_values"
                    ].unsqueeze(1)

                if options["constraint_strengths"].dim() != 2:
                    options["constraint_strengths"] = options[
                        "constraint_strengths"
                    ].unsqueeze(1)

                # Combine existing and new constraint locations
                options["constraint_locations"] = torch.cat(
                    [options["constraint_locations"], factory_locations], dim=0
                )
                options["constraint_values"] = torch.cat(
                    [options["constraint_values"], factory_values], dim=0
                )

                # Handle constraint strengths
                if options["constraint_strengths"] is None:
                    # Create constraint strengths for existing constraints using the default heuristic
                    # We need to use the original constraint_values before concatenation
                    original_values = options["constraint_values"][
                        : -factory_values.shape[0]
                    ]
                    existing_strengths = (0.2 * original_values + 0.1) ** 2
                    options["constraint_strengths"] = torch.cat(
                        [existing_strengths, factory_strengths], dim=0
                    )
                else:
                    options["constraint_strengths"] = torch.cat(
                        [options["constraint_strengths"], factory_strengths], dim=0
                    )
            else:
                # No existing constraints, just use the factory values
                options["constraint_locations"] = factory_locations
                options["constraint_values"] = factory_values
                options["constraint_strengths"] = factory_strengths

        return options
