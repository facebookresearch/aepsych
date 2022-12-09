#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from aepsych.config import Config, ConfigurableMixin
from ax.core import Experiment
from ax.modelbridge.completion_criterion import CompletionCriterion


class RunIndefinitely(CompletionCriterion, ConfigurableMixin):
    def __init__(self, run_indefinitely: bool) -> None:
        self.run_indefinitely = run_indefinitely

    def is_met(self, experiment: Experiment) -> bool:
        return not self.run_indefinitely

    @classmethod
    def get_config_options(cls, config: Config, name: str) -> Dict[str, Any]:
        run_indefinitely = config.getboolean(name, "run_indefinitely", fallback=False)
        options = {"run_indefinitely": run_indefinitely}
        return options
