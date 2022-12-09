#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from aepsych.config import Config, ConfigurableMixin
from ax.core.base_trial import TrialStatus
from ax.modelbridge.completion_criterion import MinimumTrialsInStatus


class MinAsks(MinimumTrialsInStatus, ConfigurableMixin):
    @classmethod
    def get_config_options(cls, config: Config, name: str) -> Dict[str, Any]:
        min_asks = config.getint(name, "min_asks", fallback=1)
        options = {"status": TrialStatus.RUNNING, "threshold": min_asks}
        return options
