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

AEPsychMixinType = TypeVar("AEPsychMixinType", bound=AEPsychMixin)


class AEPsychGenerator(abc.ABC, Generic[AEPsychMixinType]):
    def __init__(
        self,
    ) -> None:
        pass

    @abc.abstractmethod
    def gen(self, num_points: int, model: AEPsychMixinType) -> np.ndarray:
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Config):
        pass
