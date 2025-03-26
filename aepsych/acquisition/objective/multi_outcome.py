#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any

import torch
from aepsych.config import Config, ConfigurableMixin
from botorch.acquisition.objective import ScalarizedPosteriorTransform


class AffinePosteriorTransform(ScalarizedPosteriorTransform, ConfigurableMixin):
    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Return a dictionary of the relevant options to initialize this class from the
        config, even if it is outside of the named section. By default, this will look
        for options in name based on the __init__'s arguments/defaults.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): Primary section to look for options for this class and
                the name to infer options from other sections in the config.
            options (dict[str, Any], optional): Options to override from the config,
                defaults to None.


        Return:
            dict[str, Any]: A dictionary of options to initialize this class.
        """
        name = name or cls.__name__
        options = super().get_config_options(config, name, options)

        if "weights" not in options:
            outcomes = config.getlist("common", "outcome_types", element_type=str)
            options["weights"] = torch.ones(len(outcomes)) / len(outcomes)

        return options
