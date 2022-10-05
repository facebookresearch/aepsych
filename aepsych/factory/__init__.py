#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from ..config import Config
from .factory import (
    default_mean_covar_factory,
    monotonic_mean_covar_factory,
    ordinal_mean_covar_factory,
    song_mean_covar_factory,
)

__all__ = [
    "default_mean_covar_factory",
    "ordinal_mean_covar_factory",
    "monotonic_mean_covar_factory",
    "song_mean_covar_factory",
]

Config.register_module(sys.modules[__name__])
