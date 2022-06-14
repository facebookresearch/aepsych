#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from ..config import Config
from .gp_classification import GPClassificationModel
from .monotonic_rejection_gp import MonotonicRejectionGP
from .gp_regression import GPRegressionModel


__all__ = ["GPClassificationModel", "MonotonicRejectionGP", "GPRegressionModel"]

Config.register_module(sys.modules[__name__])
