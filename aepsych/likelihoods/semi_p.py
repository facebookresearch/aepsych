#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from aepsych.acquisition.objective import AEPsychObjective, FloorProbitObjective
from aepsych.config import Config
from gpytorch.likelihoods import _OneDimensionalLikelihood


class LinearBernoulliLikelihood(_OneDimensionalLikelihood):
    """
    A likelihood of the form Bernoulli(sigma(k(x+c))), where k and c are
    GPs and sigma is a flexible link function.
    """

    def __init__(self, objective: Optional[AEPsychObjective] = None) -> None:
        """Initializes the linear bernoulli likelihood.

        Args:
            objective (AEPsychObjective, optional): Link function to use (sigma in the notation above).
                Defaults to probit with no floor.
        """
        super().__init__()
        self.objective = objective or FloorProbitObjective(floor=0.0)

    def f(self, function_samples: torch.Tensor, Xi: torch.Tensor) -> torch.Tensor:
        """Return the latent function value, k(x-c).

        Args:
            function_samples (torch.Tensor): Samples from a batched GP
            Xi (torch.Tensor): Intensity values.

        Returns:
            torch.Tensor: latent function value.
        """
        # function_samples is of shape nsamp x (b) x 2 x n

        # If (b) is present,
        if function_samples.ndim > 3:
            assert function_samples.ndim == 4
            assert function_samples.shape[2] == 2
            # In this case, Xi will be of size b x n
            # Offset and slope should be num_samps x b x n
            offset = function_samples[:, :, 0, :]
            slope = function_samples[:, :, 1, :]
            fsamps = slope * (Xi - offset)
            # Expand from (nsamp x b x n) to (nsamp x b x n x 1)
            fsamps = fsamps.unsqueeze(-1)
        else:
            assert function_samples.ndim == 3
            assert function_samples.shape[1] == 2
            # Shape is num_samps x 2 x n
            # Offset and slope should be num_samps x n
            # Xi will be of size n
            offset = function_samples[:, 0, :]
            slope = function_samples[:, 1, :]
            fsamps = slope * (Xi - offset)
            # Expand from (nsamp x n) to (nsamp x 1 x n x 1)
            fsamps = fsamps.unsqueeze(1).unsqueeze(-1)
        return fsamps

    def p(self, function_samples: torch.Tensor, Xi: torch.Tensor) -> torch.Tensor:
        """Returns the response probability sigma(k(x+c)).

        Args:
            function_samples (torch.Tensor): Samples from the batched GP (see documentation for self.f)
            Xi (torch.Tensor): Intensity Values.

        Returns:
            torch.Tensor: Response probabilities.
        """
        fsamps = self.f(function_samples, Xi)
        return self.objective(fsamps)

    def forward(
        self, function_samples: torch.Tensor, Xi: torch.Tensor, **kwargs
    ) -> torch.distributions.Bernoulli:
        """Forward pass for the likelihood

        Args:
            function_samples (torch.Tensor): Samples from a batched GP of batch size 2.
            Xi (torch.Tensor): Intensity values.

        Returns:
            torch.distributions.Bernoulli: Outcome likelihood.
        """
        output_probs = self.p(function_samples, Xi)
        return torch.distributions.Bernoulli(probs=output_probs)

    def expected_log_prob(
        self, observations: torch.Tensor, function_dist: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """This has to be overridden to fix a bug in gpytorch where the kwargs
        aren't being passed along to self.forward.

        Args:
            observations (torch.Tensor): Observations.
            function_dist (torch.Tensor): Function distribution.


        Returns:
            torch.Tensor: Expected log probability.
        """

        # modified, TODO fixme upstream (cc @bletham)
        def log_prob_lambda(function_samples: torch.Tensor) -> torch.Tensor:
            """Lambda function to compute the log probability.

            Args:
                function_samples (torch.Tensor): Function samples.

            Returns:
                torch.Tensor: Log probability.
            """
            return self.forward(function_samples, **kwargs).log_prob(observations)

        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob

    @classmethod
    def from_config(cls, config: Config) -> "LinearBernoulliLikelihood":
        """Create an instance from a configuration object.

        Args:
            config (Config): Configuration object.

        Returns:
            LinearBernoulliLikelihood: LinearBernoulliLikelihood instance.
        """
        classname = cls.__name__

        objective = config.getobj(classname, "objective")

        if hasattr(objective, "from_config"):
            objective = objective.from_config(config)
        else:
            objective = objective

        return cls(objective=objective)
