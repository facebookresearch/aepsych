#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional, Tuple

from ax.models.torch.botorch_modular.acquisition import Acquisition
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform


class AEPsychAcquisition(Acquisition):
    """Acquisition functions use the strategy's model to determine which points should be sampled next,
    with some overarching goal in mind. We recommend PairwiseMCPosteriorVariance for global exploration,
    and qNoisyExpectedImprovement for optimization. For other options, check out the botorch and aepsych docs.
    The class also subsets the model to include only the outcomes needed for optimization.
    If multiple objectives are present, the class will also infer the objective thresholds using the model."""

    def get_botorch_objective_and_transform(
        self, **kwargs
    ) -> Tuple[Optional[MCAcquisitionObjective], Optional[PosteriorTransform]]:

        objective, transform = super().get_botorch_objective_and_transform(**kwargs)

        if "objective" in self.options:
            objective = self.options.pop("objective")

        return objective, transform
