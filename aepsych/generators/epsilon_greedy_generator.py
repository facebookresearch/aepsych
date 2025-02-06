#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import numpy as np
import torch
from aepsych.models.base import AEPsychModelMixin

from .base import AEPsychGenerator
from .optimize_acqf_generator import OptimizeAcqfGenerator


class EpsilonGreedyGenerator(AEPsychGenerator):
    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        subgenerator: AEPsychGenerator,
        epsilon: float = 0.1,
    ) -> None:
        """Initialize EpsilonGreedyGenerator.

        Args:
            lb (torch.Tensor): Lower bounds for the optimization.
            ub (torch.Tensor): Upper bounds for the optimization.
            subgenerator (AEPsychGenerator): The generator to use when not exploiting.
            epsilon (float): The probability of exploration. Defaults to 0.1.
        """
        self.subgenerator = subgenerator
        self.epsilon = epsilon
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

    @classmethod
    def get_config_options(
        cls,
        config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Find the config options for the generator.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): Unused, kept for API conformity.
            options (Dict[str, Any], optional): Existing options, any key in options
                will be ignored from the config.

        Return:
            Dict[str, Any]: A dictionary of options to initialize the generator.
        """
        options = super().get_config_options(config, name, options)

        if "subgenerator" not in options:  # Missing subgenerator
            options["subgenerator"] = OptimizeAcqfGenerator.from_config(config)

        return options

    def gen(
        self,
        num_points: int,
        model: AEPsychModelMixin,
        fixed_features: Optional[Dict[int, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Query next point(s) to run by sampling from the subgenerator with probability 1-epsilon, and randomly otherwise.

        Args:
            num_points (int): Number of points to query.
            model (AEPsychModelMixin): Model to use for generating points.
            fixed_features: (Dict[int, float], optional): Parameters that are fixed to specific values.
            **kwargs: Passed to subgenerator if not exploring
        """
        if num_points > 1:
            raise NotImplementedError("Epsilon-greedy batched gen is not implemented!")
        if np.random.uniform() < self.epsilon:
            sample_ = np.random.uniform(low=self.lb, high=self.ub)
            sample = torch.tensor(sample_).reshape(1, -1)

            if fixed_features is not None:
                for key, value in fixed_features.items():
                    sample[:, key] = value

            return sample
        else:
            return self.subgenerator.gen(
                num_points, model, fixed_features=fixed_features, **kwargs
            )
