#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc
from inspect import signature
from typing import Any, Generic, Protocol, runtime_checkable, TypeVar

import numpy as np
from aepsych.config import Config
from aepsych.models.base import AEPsychMixin
from botorch.acquisition import (
    AcquisitionFunction,
    NoisyExpectedImprovement,
    qNoisyExpectedImprovement,
)

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

    def __init__(
        self,
    ) -> None:
        pass

    @abc.abstractmethod
    def gen(self, num_points: int, model: AEPsychModelType) -> np.ndarray:
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Config):
        pass

    @classmethod
    def _get_acqf_options(cls, acqf: AcquisitionFunction, config: Config):
        if acqf is not None:
            acqf_name = acqf.__name__
            default_extra_acqf_args = {
                "beta": 3.84,
                "target": 0.75,
                "objective": None,
                "query_set_size": 512,
                "posterior_transform": None,
            }
            extra_acqf_args = {
                k: config.getobj(
                    acqf_name, k, fallback_type=float, fallback=v, warn=False
                )
                for k, v in default_extra_acqf_args.items()
            }
            acqf_args_expected = signature(acqf).parameters.keys()
            extra_acqf_args = {
                k: v for k, v in extra_acqf_args.items() if k in acqf_args_expected
            }
            for k, v in extra_acqf_args.items():
                if hasattr(v, "from_config"):  # configure if needed
                    assert isinstance(v, AcqArgProtocol)  # make mypy happy
                    extra_acqf_args[k] = v.from_config(config)
                elif isinstance(v, type):  # instaniate a class if needed
                    extra_acqf_args[k] = v()
        else:
            extra_acqf_args = {}

        return extra_acqf_args
