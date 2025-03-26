#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch
from aepsych.config import ConfigurableMixin
from aepsych.likelihoods import LinearBernoulliLikelihood
from botorch.acquisition.objective import MCAcquisitionObjective
from gpytorch.likelihoods import Likelihood
from torch import Tensor


class SemiPObjectiveBase(MCAcquisitionObjective, ConfigurableMixin):
    """Wraps the semi-parametric transform into an objective
    that correctly extracts various things
    """

    # because we have an extra dim for the SemiP batch dimension,
    # all the q-batch output shape checks fail, disable them here
    _verify_output_shape: bool = False

    def __init__(self, stim_dim: int = 0) -> None:
        """Initialize the SemiPObjectiveBase.

        Args:
            stim_dim (int): The stimulus dimension. Defaults to 0.
        """
        super().__init__()
        self.stim_dim = stim_dim


class SemiPProbabilityObjective(SemiPObjectiveBase):
    """Wraps the semi-parametric transform into an objective
    that gives outcome probabilities
    """

    def __init__(self, likelihood: Likelihood = None, *args, **kwargs):
        """Evaluates the probability objective.

        Args:
            likelihood (Likelihood). Underlying SemiP likelihood (which we use for its objective/link)
            other arguments are passed to the base class (notably, stim_dim).
        """
        super().__init__(*args, **kwargs)
        self.likelihood = likelihood or LinearBernoulliLikelihood()

    def forward(self, samples: Tensor, X: Tensor) -> Tensor:
        """Evaluates the probability objective.

        Args:
            samples (Tensor): GP samples.
            X (Tensor): Inputs at which to evaluate objective. Unlike most AEPsych objectives,
                we need X here to split out the intensity dimension.

        Returns:
            Tensor: Response probabilities at the specific X values and function samples.
        """

        Xi = X[..., self.stim_dim]
        # the output of LinearBernoulliLikelihood is (nsamp x b x n x 1)
        # but the output of MCAcquisitionObjective should be `nsamp x *batch_shape x q`
        # so we remove the final dim
        return self.likelihood.p(function_samples=samples, Xi=Xi).squeeze(-1)

    @classmethod
    def get_config_options(
        cls,
        config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Find the config options for the objective.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): Unused, kept for API conformity.
            options (dict[str, Any], optional): Existing options, any key in options
                will be ignored from the config.

        Return:
            dict[str, Any]: A dictionary of options to initialize the objective.
        """
        options = super().get_config_options(config, name, options)

        # Due to likelihood object not having a from_config, so we just initialize it
        if isinstance(options["likelihood"], type):
            options["likelihood"] = options["likelihood"]()

        return options


class SemiPThresholdObjective(SemiPObjectiveBase):
    """Wraps the semi-parametric transform into an objective
    that gives the threshold distribution.
    """

    def __init__(
        self,
        target: float = 0.75,
        likelihood: LinearBernoulliLikelihood | None = None,
        *args,
        **kwargs,
    ):
        """Evaluates the probability objective.

        Args:
            target (float): the threshold to evaluate.
            likelihood (LinearBernoulliLikelihood, optional): Underlying SemiP likelihood (which we use for its inverse link). Defaults to None.

            other arguments are passed to the base class (notably, stim_dim).
        """
        super().__init__(*args, **kwargs)

        self.likelihood = likelihood or LinearBernoulliLikelihood()
        self.fspace_target = self.likelihood.objective.inverse(torch.tensor(target))

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        """Evaluates the probability objective.

        Args:
            samples (Tensor): GP samples.
            X (Tensor, optional): Ignored, here for compatibility with the objective API.

        Returns:
            Tensor: Threshold probabilities at the specific GP sample values.
        """
        offset = samples[..., 0, :]
        slope = samples[..., 1, :]
        return (self.fspace_target + slope * offset) / slope

    @classmethod
    def get_config_options(
        cls,
        config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Find the config options for the objective.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): Unused, kept for API conformity.
            options (dict[str, Any], optional): Existing options, any key in options
                will be ignored from the config.

        Return:
            dict[str, Any]: A dictionary of options to initialize the objective.
        """
        options = super().get_config_options(config, name, options)

        # Due to likelihood object not having a from_config, so we just initialize it
        if isinstance(options["likelihood"], type):
            options["likelihood"] = options["likelihood"]()

        return options
