# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Optional

from botorch.acquisition.objective import MCAcquisitionObjective
from torch import Tensor


class AEPsychObjective(MCAcquisitionObjective):
    def inverse(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError
