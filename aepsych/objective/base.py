# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Any, Optional, Tuple

import torch
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.posteriors import GPyTorchPosterior
from torch import Tensor


class AEPsychObjective(MCAcquisitionObjective):
    def inverse(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError

    def posterior_transform(
        self, posterior: GPyTorchPosterior, **kwargs: Any
    ) -> Tuple[Tensor, Tensor]:
        """Evaluates the objective on a posterior, returning mean and variance.
        This base implementation uses Monte Carlo sampling, but some objectives may have analytic solutions.

        Args:
            posterior (botorch.posteriors.GPyTorchPosterior): A posterior evaluated at some points.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed posterior mean and variance.
        """

        samps = kwargs.get("samples", 10000)
        fsamps = posterior.sample(torch.Size([samps]))
        psamps = self.forward(fsamps)
        pmean, pvar = psamps.mean(0), psamps.var(0)
        return pmean, pvar
