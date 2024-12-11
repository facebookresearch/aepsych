#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from ..config import Config
from .gp_classification import GPBetaRegressionModel, GPClassificationModel
from .gp_regression import GPRegressionModel
from .monotonic_projection_gp import MonotonicProjectionGP
from .monotonic_rejection_gp import MonotonicRejectionGP
from .multitask_regression import IndependentMultitaskGPRModel, MultitaskGPRModel
from .ordinal_gp import OrdinalGPModel
from .pairwise_probit import PairwiseProbitModel
from .semi_p import (
    HadamardSemiPModel,
    semi_p_posterior_transform,
    SemiParametricGPModel,
)

__all__ = [
    "GPClassificationModel",
    "MonotonicRejectionGP",
    "GPRegressionModel",
    "OrdinalGPModel",
    "MonotonicProjectionGP",
    "MultitaskGPRModel",
    "IndependentMultitaskGPRModel",
    "HadamardSemiPModel",
    "SemiParametricGPModel",
    "semi_p_posterior_transform",
    "GPBetaRegressionModel",
    "PairwiseProbitModel",
]

Config.register_module(sys.modules[__name__])
