#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from inspect import signature
from aepsych.config import Config

from aepsych.models.base import ModelProtocol
from botorch.acquisition import (
    AcquisitionFunction,
    NoisyExpectedImprovement,
    qNoisyExpectedImprovement,
)
import torch
from botorch.optim import optimize_acqf
import numpy as np
from aepsych.generators.base import AEPsychGenerator
from typing import Dict, Optional, Any


def _prune_extra_acqf_args(acqf, extra_acqf_args: Dict):
    # prune extra args needed, ignore the rest
    # (this helps with API consistency)
    acqf_args_expected = signature(acqf).parameters.keys()
    return {k: v for k, v in extra_acqf_args.items() if k in acqf_args_expected}


class OptimizeAcqfGenerator(AEPsychGenerator):
    """Generator that chooses points by minimizing an acquisition function."""

    baseline_requiring_acqfs = [qNoisyExpectedImprovement, NoisyExpectedImprovement]

    def __init__(
        self,
        acqf: AcquisitionFunction,
        acqf_kwargs: Optional[Dict[str, Any]] = None,
        restarts: int = 10,
        samps: int = 1000,
    ) -> None:
        """Initialize OptimizeAcqfGenerator.
        Args:
            acqf (AcquisitionFunction): Acquisition function to use.
            acqf_kwargs (Dict[str, object], optional): Extra arguments to
                pass to acquisition function. Defaults to no arguments.
            restarts (int): Number of restarts for acquisition function optimization.
            samps (int): Number of samples for quasi-random initialization of the acquisition function optimizer.
        """
        if acqf_kwargs is None:
            acqf_kwargs = {}
        self.acqf = acqf
        self.acqf_kwargs = acqf_kwargs
        self.restarts = restarts
        self.samps = samps

    def _instantiate_acquisition_fn(self, model: ModelProtocol, train_x):
        if self.acqf in self.baseline_requiring_acqfs:
            return self.acqf(model=model, X_baseline=train_x, **self.acqf_kwargs)
        else:
            return self.acqf(model=model, **self.acqf_kwargs)

    def gen(self, num_points: int, model: ModelProtocol) -> np.ndarray:
        """Query next point(s) to run by optimizing the acquisition function.
        Args:
            num_points (int, optional): Number of points to query.
            model (ModelProtocol): Fitted model of the data.
        Returns:
            np.ndarray: Next set of point(s) to evaluate, [num_points x dim].
        """

        # eval should be inherited from superclass
        model.eval()  # type: ignore
        train_x = model.train_inputs[0]
        acqf = self._instantiate_acquisition_fn(model, train_x)

        new_candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=torch.tensor(np.c_[model.lb, model.ub]).T.to(train_x),
            q=num_points,
            num_restarts=self.restarts,
            raw_samples=self.samps,
        )

        return new_candidate.numpy()

    @classmethod
    def from_config(cls, config: Config):
        classname = cls.__name__
        acqf = config.getobj("experiment", "acqf")
        acqf_name = acqf.__name__

        default_extra_acqf_args = {
            "beta": 3.98,
            "target": 0.75,
            "objective": None,
        }
        extra_acqf_args = {
            k: config.getobj(acqf_name, k, fallback_type=float, fallback=v, warn=False)
            for k, v in default_extra_acqf_args.items()
        }
        extra_acqf_args = _prune_extra_acqf_args(acqf, extra_acqf_args)
        if (
            "objective" in extra_acqf_args.keys()
            and extra_acqf_args["objective"] is not None
        ):
            extra_acqf_args["objective"] = extra_acqf_args["objective"]()

        restarts = config.getint(classname, "restarts", fallback=10)
        samps = config.getint(classname, "samps", fallback=1000)

        return cls(acqf, extra_acqf_args, restarts, samps)
