#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable

import torch
from aepsych.config import ConfigurableMixin
from gpytorch.likelihoods import _OneDimensionalLikelihood


class BernoulliObjectiveLikelihood(_OneDimensionalLikelihood, ConfigurableMixin):
    """
    Bernoulli likelihood with a flexible link (objective) defined
    by a callable (which can be a botorch objective)
    """

    def __init__(self, objective: Callable) -> None:
        """Initialize BernoulliObjectiveLikelihood.

        Args:
            objective (Callable): Objective function that maps function samples to probabilities.
        """
        super().__init__()
        self.objective = objective

    def forward(
        self, function_samples: torch.Tensor, **kwargs: Any
    ) -> torch.distributions.Bernoulli:
        """Forward pass for BernoulliObjectiveLikelihood.

        Args:
            function_samples (torch.Tensor): Function samples.

        Returns:
            torch.distributions.Bernoulli: Bernoulli distribution object.
        """
        output_probs = self.objective(function_samples)
        return torch.distributions.Bernoulli(probs=output_probs)

    @classmethod
    def get_config_options(
        cls,
        config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Find the config options for the likelihood.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): Unused, kept for API conformity.
            options (dict[str, Any], optional): Existing options, any key in options
                will be ignored from the config.

        Return:
            dict[str, Any]: A dictionary of options to initialize the likelihood.
        """
        options = super().get_config_options(config, name, options)
        options["objective"] = options["objective"].from_config(config)
        return options
