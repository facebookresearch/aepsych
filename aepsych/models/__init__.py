#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from ..config import Config
from .gp_classification import GPClassificationModel
from .gp_regression import GPRegressionModel
from .independent_gps import IndependentGPsModel
from .monotonic_projection_gp import MonotonicProjectionGP
from .ordinal_gp import OrdinalGPModel
from .pairwise_probit import PairwiseProbitModel
from .semi_p import (
    HadamardSemiPModel,
    semi_p_posterior_transform,
    SemiParametricGPModel,
)
from .utils import constraint_factory
from .variational_gp import VariationalGPModel

__all__ = [
    "GPClassificationModel",
    "GPRegressionModel",
    "OrdinalGPModel",
    "MonotonicProjectionGP",
    "HadamardSemiPModel",
    "SemiParametricGPModel",
    "semi_p_posterior_transform",
    "PairwiseProbitModel",
    "VariationalGPModel",
    "IndependentGPsModel",
]

Config.register_module(sys.modules[__name__])
Config.register_object(constraint_factory)
