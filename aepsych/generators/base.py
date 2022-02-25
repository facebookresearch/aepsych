#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc
from aepsych.config import Config
from aepsych.models.base import AEPsychMixin
import numpy as np
from typing import Generic, TypeVar
from botorch.acquisition import (
    AcquisitionFunction,
    qNoisyExpectedImprovement,
    NoisyExpectedImprovement,
)
from inspect import signature

AEPsychModelType = TypeVar("AEPsychModelType", bound=AEPsychMixin)


class AEPsychGenerator(abc.ABC, Generic[AEPsychModelType]):
    """Abstract base class for generators, which are responsible for generating new points to sample."""

    _requires_model = True
    baseline_requiring_acqfs = [qNoisyExpectedImprovement, NoisyExpectedImprovement]

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
                "beta": 3.98,
                "target": 0.75,
                "objective": None,
                "query_set_size": 512,
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
            if (
                "objective" in extra_acqf_args.keys()
                and extra_acqf_args["objective"] is not None
            ):
                extra_acqf_args["objective"] = extra_acqf_args["objective"]()
        else:
            extra_acqf_args = {}

        return extra_acqf_args
