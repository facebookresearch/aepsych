#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from ..config import Config
from .lookahead import ApproxGlobalSUR, EAVC, GlobalMI, GlobalSUR, LocalMI, LocalSUR
from .lse import MCLevelSetEstimation
from .mc_posterior_variance import MCPosteriorVariance, MonotonicMCPosteriorVariance
from .monotonic_rejection import MonotonicMCLSE
from .mutual_information import (
    BernoulliMCMutualInformation,
    MonotonicBernoulliMCMutualInformation,
)
from .objective import (
    FloorGumbelObjective,
    FloorLogitObjective,
    FloorProbitObjective,
    ProbitObjective,
    semi_p,
)

lse_acqfs = [
    MonotonicMCLSE,
    GlobalMI,
    GlobalSUR,
    ApproxGlobalSUR,
    EAVC,
    LocalMI,
    LocalSUR,
]
__all__ = [
    "BernoulliMCMutualInformation",
    "MonotonicBernoulliMCMutualInformation",
    "MonotonicMCLSE",
    "MCPosteriorVariance",
    "MonotonicMCPosteriorVariance",
    "MCPosteriorVariance",
    "MCLevelSetEstimation",
    "ProbitObjective",
    "FloorProbitObjective",
    "FloorLogitObjective",
    "FloorGumbelObjective",
    "GlobalMI",
    "GlobalSUR",
    "ApproxGlobalSUR",
    "EAVC",
    "LocalMI",
    "LocalSUR",
    "semi_p",
]

Config.register_module(sys.modules[__name__])
