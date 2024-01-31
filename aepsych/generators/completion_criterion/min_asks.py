#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Set

from aepsych.config import Config, ConfigurableMixin

from ax.core.experiment import Experiment
from ax.modelbridge.transition_criterion import TransitionCriterion


class MinAsks(TransitionCriterion, ConfigurableMixin):
    def __init__(
        self,
        threshold: int,
        block_transition_if_unmet: Optional[bool] = True,
        block_gen_if_met: Optional[bool] = False,
    ) -> None:
        self.threshold = threshold
        super().__init__(
            block_transition_if_unmet=block_transition_if_unmet,
            block_gen_if_met=block_gen_if_met,
        )

    def is_met(
        self, experiment: Experiment, trials_from_node: Optional[Set[int]] = None
    ) -> bool:
        return experiment.num_asks >= self.threshold

    def block_continued_generation_error(
        self,
        node_name: Optional[str],
        model_name: Optional[str],
        experiment: Optional[Experiment],
        trials_from_node: Optional[Set[int]] = None,
    ) -> None:
        pass

    @classmethod
    def get_config_options(cls, config: Config, name: str) -> Dict[str, Any]:
        min_asks = config.getint(name, "min_asks", fallback=1)
        options = {
            "threshold": min_asks,
        }
        return options
