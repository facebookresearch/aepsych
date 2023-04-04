#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from aepsych.models.base import AEPsychModel
from aepsych.utils import promote_0d
from aepsych.utils_logging import getLogger

logger = getLogger()


class ExactGP(AEPsychModel, SingleTaskGP):
    @classmethod
    def get_mll_class(cls):
        return ExactMarginalLogLikelihood

    def predict(
        self, x: Union[torch.Tensor, np.ndarray], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Posterior mean and variance at queries points.
        """
        with torch.no_grad():
            post = self.posterior(x)
        fmean = post.mean.squeeze()
        fvar = post.variance.squeeze()
        return promote_0d(fmean), promote_0d(fvar)


class ContinuousRegressionGP(ExactGP):
    """GP Regression model for single continuous outcomes, using exact inference."""

    stimuli_per_trial = 1
    outcome_type = "continuous"
