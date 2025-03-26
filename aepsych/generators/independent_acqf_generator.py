#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

import torch
from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.models import IndependentGPsModel
from aepsych.utils_logging import getLogger

logger = getLogger()


class IndependentOptimizeAcqfGenerator(AEPsychGenerator):
    """Generator that is multiple OptimizeAcqfGenerator optimized sequentially. Intended
    to be used with IndependentGPsModel for multioutcome experiments."""

    _requires_model = True
    stimuli_per_trial = 1
    max_asks: int | None = None
    dim: int

    def __init__(
        self, generators: list[AEPsychGenerator], lb: torch.Tensor, ub: torch.Tensor
    ) -> None:
        # Validate the generators all have commensurate attributes
        self.stimuli_per_trial = generators[0].stimuli_per_trial
        if not all(
            [gen.stimuli_per_trial == self.stimuli_per_trial for gen in generators[1:]]
        ):
            raise ValueError("Not every generator has the same stimuli_per_trial")

        self.max_asks = generators[0].max_asks
        if not all([gen.max_asks == self.max_asks for gen in generators[1:]]):
            raise ValueError("Not every generator has the same max_asks")

        self.dim = lb.shape[0]
        if not all([gen.dim == self.dim for gen in generators[1:]]):
            raise ValueError("Not every generator match dimensions")

        self.generators = generators
        self.lb = lb
        self.ub = ub

        # Check that each generator has the matching bounds, if not fix it
        # important for bounded generators like OptimizeAcqfGenerators
        for generator in self.generators:
            if not hasattr(generator, "lb") or not hasattr(generator, "ub"):
                continue

            if torch.equal(generator.lb, self.lb):
                generator.lb = self.lb

            if torch.equal(generator.ub, self.ub):
                generator.ub = self.ub

    def gen(
        self,
        num_points: int,
        model: IndependentGPsModel,
        fixed_features: dict[int, float] | None = None,
        order: list[int] | torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if order is None:
            order = torch.arange(0, len(self.generators))
            order = order[torch.randperm(order.nelement())]

        points = torch.empty((0, self.dim))
        if model is not None:
            points = points.to(model.device)

        for i in order:
            point = self.generators[i].gen(
                num_points=num_points,
                model=model[int(i)],
                fixed_features=fixed_features,
                X_pending=points if points.shape[0] > 0 else None,
                **kwargs,
            )
            points = torch.concatenate((points, point))

        return points

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Return a dictionary of the relevant options to initialize the
        IndependentOptimizeAcqfGenerator. Primarily, this creates all the necessary
        generators to populate the generator list.

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
        options = super().get_config_options(config=config, name=name, options=options)

        if "generators" not in options:
            generators = []
            generator_names = config.getlist(name, "generators", element_type=str)
            for gen_name in generator_names:
                if gen_name in Config.registered_names:
                    gen_cls = Config.registered_names[gen_name]
                else:  # Aliased class
                    gen_cls = config.getobj(gen_name, "class")
                if not hasattr(gen_cls, "from_config"):
                    raise ValueError(
                        f"IndependentOptimizeAcqfGenerator was given a generator {gen_cls} that cannot be configured."
                    )
                generators.append(gen_cls.from_config(config, gen_name))
            options["generators"] = generators

        return options
