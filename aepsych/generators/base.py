#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc
from inspect import signature
from typing import Any, Dict, Generic, Protocol, runtime_checkable, TypeVar, Optional

import torch
from aepsych.config import Config, ConfigurableMixin
from aepsych.models.base import AEPsychMixin
from ax.core.experiment import Experiment
from ax.modelbridge.generation_node import GenerationStep
from botorch.acquisition import (
    AcquisitionFunction,
    NoisyExpectedImprovement,
    qNoisyExpectedImprovement,
)

from .completion_criterion import completion_criteria

AEPsychModelType = TypeVar("AEPsychModelType", bound=AEPsychMixin)


@runtime_checkable
class AcqArgProtocol(Protocol):
    @classmethod
    def from_config(cls, config: Config) -> Any:
        pass


class AEPsychGenerator(abc.ABC, Generic[AEPsychModelType]):
    """Abstract base class for generators, which are responsible for generating new points to sample."""

    _requires_model = True
    baseline_requiring_acqfs = [qNoisyExpectedImprovement, NoisyExpectedImprovement]
    stimuli_per_trial = 1
    max_asks: Optional[int] = None

    def __init__(
        self,
    ) -> None:
        pass

    @abc.abstractmethod
    def gen(self, num_points: int, model: AEPsychModelType) -> torch.Tensor:
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Config):
        pass

    @classmethod
    def _get_acqf_options(cls, acqf: AcquisitionFunction, config: Config):
        if acqf is not None:
            acqf_name = acqf.__name__

            # model is not an extra arg, it's a default arg
            acqf_args_expected = [
                i for i in list(signature(acqf).parameters.keys()) if i != "model"
            ]

            # this is still very ugly
            extra_acqf_args = {}
            if acqf_name in config:
                full_section = config[acqf_name]
                for k in acqf_args_expected:
                    # if this thing is configured
                    if k in full_section.keys():
                        # if it's an object make it an object
                        if full_section[k] in Config.registered_names.keys():
                            extra_acqf_args[k] = config.getobj(acqf_name, k)
                        else:
                            # otherwise try a float
                            try:
                                extra_acqf_args[k] = config.getfloat(acqf_name, k)
                            # finally just return a string
                            except ValueError:
                                extra_acqf_args[k] = config.get(acqf_name, k)

            # next, do more processing
            for k, v in extra_acqf_args.items():
                if hasattr(v, "from_config"):  # configure if needed
                    assert isinstance(v, AcqArgProtocol)  # make mypy happy
                    extra_acqf_args[k] = v.from_config(config)
                elif isinstance(v, type):  # instaniate a class if needed
                    extra_acqf_args[k] = v()
        else:
            extra_acqf_args = {}

        return extra_acqf_args


class AEPsychGenerationStep(GenerationStep, ConfigurableMixin, abc.ABC):
    def __init__(self, name, **kwargs):
        super().__init__(num_trials=-1, **kwargs)
        self.name = name

    @classmethod
    def get_config_options(cls, config: Config, name: str) -> Dict:
        criteria = []
        for crit in completion_criteria:
            # TODO: Figure out how to convince mypy that CompletionCriterion have `from_config`
            criterion = crit.from_config(config, name)  # type: ignore
            criteria.append(criterion)
        options = {"completion_criteria": criteria, "name": name}
        return options

    def finished(self, experiment: Experiment):
        finished = all(
            [criterion.is_met(experiment) for criterion in self.completion_criteria]
        )
        return finished
