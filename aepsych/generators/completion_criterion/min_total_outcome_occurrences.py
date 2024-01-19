#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from aepsych.config import Config, ConfigurableMixin

from ax.modelbridge.transition_criterion import MinimumPreferenceOccurances


class MinTotalOutcomeOccurrences(MinimumPreferenceOccurances, ConfigurableMixin):
    @classmethod
    def get_config_options(cls, config: Config, name: str) -> Dict[str, Any]:
        outcome_types = config.getlist(name, "outcome_types", element_type=str)
        outcome_names = config.getlist(
            name, "outcome_names", element_type=str, fallback=None
        )
        # The completion criterion needs to get the name of the first outcome.
        # TODO: Make it so that the criterion can be configured to which outcome
        # it cares about instead of defaulting to the first one.
        if outcome_names is None:
            outcome_name = "outcome_1"
        else:
            outcome_name = str(outcome_names[0])
        min_total_outcome_occurrences = config.getint(
            name,
            "min_total_outcome_occurrences",
            fallback=1 if "binary" in outcome_types else 0,
        )
        options = {
            "metric_name": outcome_name,
            "threshold": min_total_outcome_occurrences,
        }
        return options
