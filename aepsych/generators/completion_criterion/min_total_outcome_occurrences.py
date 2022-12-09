#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from aepsych.config import Config, ConfigurableMixin
from ax.modelbridge.completion_criterion import MinimumPreferenceOccurances


class MinTotalOutcomeOccurrences(MinimumPreferenceOccurances, ConfigurableMixin):
    @classmethod
    def get_config_options(cls, config: Config, name: str) -> Dict[str, Any]:
        outcome_types = config.getlist(name, "outcome_types", element_type=str)
        min_total_outcome_occurrences = config.getint(
            name,
            "min_total_outcome_occurrences",
            fallback=1 if "binary" in outcome_types else 0,
        )
        options = {
            "metric_name": "objective",
            "threshold": min_total_outcome_occurrences,
        }
        return options
