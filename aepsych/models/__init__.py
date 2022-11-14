#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from ..config import Config
from .gp_classification import GPClassificationModel
from .gp_regression import GPRegressionModel
from .monotonic_projection_gp import MonotonicProjectionGP
from .monotonic_rejection_gp import MonotonicRejectionGP
from .ordinal_gp import OrdinalGPModel
from .pairwise_probit import PairwiseProbitModel

__all__ = [
    "GPClassificationModel",
    "MonotonicRejectionGP",
    "GPRegressionModel",
    "PairwiseProbitModel",
    "OrdinalGPModel",
    "MonotonicProjectionGP",
]

Config.register_module(sys.modules[__name__])
