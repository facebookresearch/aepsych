#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from aepsych.models.base import AEPsychModel
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood


class ExactGP(AEPsychModel, SingleTaskGP):
    @classmethod
    def get_mll_class(cls):
        return ExactMarginalLogLikelihood


class ContinuousRegressionGP(ExactGP):
    """GP Regression model for single continuous outcomes, using exact inference."""

    stimuli_per_trial = 1
    outcome_type = "continuous"
