#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from ..config import Config
from .monotonic import MonotonicSingleProbitModelbridge
from .single_probit import (
    SingleProbitModelbridge,
    SingleProbitModelbridgeWithSongHeuristic,
)

__all__ = [
    "MonotonicSingleProbitModelbridge",
    "SingleProbitModelbridge",
    "SingleProbitModelbridgeWithSongHeuristic",
]

Config.register_module(sys.modules[__name__])
