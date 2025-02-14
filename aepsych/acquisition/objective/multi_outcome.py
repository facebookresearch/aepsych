#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from botorch.acquisition.objective import ScalarizedPosteriorTransform
from aepsych.config import ConfigurableMixin

class AffinePosteriorTransform(ScalarizedPosteriorTransform, ConfigurableMixin):
    pass