#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any, Dict, Optional, Tuple

import torch
from aepsych.config import Config
from aepsych.generators.base import AcqfGenerator, AEPsychGenerator
from aepsych.generators.sobol_generator import SobolGenerator
from aepsych.models.base import AEPsychModelMixin
from aepsych.utils_logging import getLogger
from botorch.acquisition import AcquisitionFunction

logger = getLogger()


class GridEvalAcqfGenerator(AcqfGenerator):
    """Abstract base class for generators that evaluate acquisition functions along a grid."""

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        acqf: AcquisitionFunction,
        acqf_kwargs: Optional[Dict[str, Any]] = None,
        samps: int = 1024,
        stimuli_per_trial: int = 1,
        grid_generator: Optional[AEPsychGenerator] = None,
    ) -> None:
        """Initialize GridEvalAcqfGenerator.
        Args:
            lb (torch.Tensor): Lower bounds for the optimization.
            ub (torch.Tensor): Upper bounds for the optimization.
            acqf (AcquisitionFunction): Acquisition function to use.
            acqf_kwargs (Dict[str, object], optional): Extra arguments to
                pass to acquisition function. Defaults to no arguments.
            samps (int): Number of quasi-random samples to evaluate acquisition function on. Defaults to 1000.
            stimuli_per_trial (int): Number of stimuli per trial. Defaults to 1.
        """
        dim = len(lb)
        self.grid_gen = grid_generator or SobolGenerator(lb, ub, dim, stimuli_per_trial)

        if acqf_kwargs is None:
            acqf_kwargs = {}
        self.acqf = acqf
        self.acqf_kwargs = acqf_kwargs
        self.samps = samps

    def gen(
        self,
        num_points: int,
        model: AEPsychModelMixin,
        fixed_features: Optional[Dict[int, float]] = None,
        **gen_options,
    ) -> torch.Tensor:
        """Query next point(s) to run by optimizing the acquisition function.
        Args:
            num_points (int): Number of points to query.
            model (AEPsychModelMixin): Fitted model of the data.
        Returns:
            torch.Tensor: Next set of point(s) to evaluate, [num_points x dim].
        """

        if self.stimuli_per_trial == 2:
            qbatch_points = self._gen(
                num_points=num_points * 2,
                model=model,
                fixed_features=fixed_features,
                **gen_options,
            )

            # output of super() is (q, dim) but the contract is (num_points, dim, 2)
            # so we need to split q into q and pairs and then move the pair dim to the end
            return qbatch_points.reshape(num_points, 2, -1).swapaxes(-1, -2)

        else:
            return self._gen(
                num_points=num_points,
                model=model,
                fixed_features=fixed_features,
                **gen_options,
            )

    @abc.abstractmethod
    def _gen(
        self,
        num_points: int,
        model: AEPsychModelMixin,
        fixed_features: Optional[Dict[int, float]] = None,
        **gen_options,
    ) -> torch.Tensor:
        pass

    def _eval_acqf(
        self,
        num_points: int,
        model: AEPsychModelMixin,
        fixed_features: Optional[Dict[int, float]] = None,
        **gen_options,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # eval should be inherited from superclass
        model.eval()  # type: ignore
        acqf = self._instantiate_acquisition_fn(model)
        X_rnd = self.grid_gen.gen(num_points, model, fixed_features, **gen_options)
        if len(X_rnd.shape) < 3:
            X_rnd = X_rnd.unsqueeze(
                1
            )  # need q in X_rnd so acqf will return proper shape
        acqf_vals = acqf(X_rnd).to(torch.float64)
        return X_rnd, acqf_vals
