#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import inspect
import time
from typing import Any, Dict, Optional

import torch
from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.models.base import ModelProtocol
from aepsych.utils_logging import getLogger
from botorch.acquisition import (
    AcquisitionFunction,
    LogNoisyExpectedImprovement,
    NoisyExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf

logger = getLogger()


class OptimizeAcqfGenerator(AEPsychGenerator):
    """Generator that chooses points by minimizing an acquisition function."""

    baseline_requiring_acqfs = [
        NoisyExpectedImprovement,
        LogNoisyExpectedImprovement,
        qNoisyExpectedImprovement,
        qLogNoisyExpectedImprovement,
    ]

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        acqf: AcquisitionFunction,
        acqf_kwargs: Optional[Dict[str, Any]] = None,
        restarts: int = 10,
        samps: int = 1000,
        max_gen_time: Optional[float] = None,
        stimuli_per_trial: int = 1,
    ) -> None:
        """Initialize OptimizeAcqfGenerator.
        Args:
            lb (torch.Tensor): Lower bounds for the optimization.
            ub (torch.Tensor): Upper bounds for the optimization.
            acqf (AcquisitionFunction): Acquisition function to use.
            acqf_kwargs (Dict[str, object], optional): Extra arguments to
                pass to acquisition function. Defaults to no arguments.
            restarts (int): Number of restarts for acquisition function optimization. Defaults to 10.
            samps (int): Number of samples for quasi-random initialization of the acquisition function optimizer. Defaults to 1000.
            max_gen_time (float, optional): Maximum time (in seconds) to optimize the acquisition function. Defaults to None.
            stimuli_per_trial (int): Number of stimuli per trial. Defaults to 1.
        """

        if acqf_kwargs is None:
            acqf_kwargs = {}
        self.acqf = acqf
        self.acqf_kwargs = acqf_kwargs
        self.restarts = restarts
        self.samps = samps
        self.max_gen_time = max_gen_time
        self.stimuli_per_trial = stimuli_per_trial
        self.lb = lb
        self.ub = ub

    def _instantiate_acquisition_fn(self, model: ModelProtocol) -> AcquisitionFunction:
        """
        Instantiates the acquisition function with the specified model and additional arguments.

        Args:
            model (ModelProtocol): The model to use with the acquisition function.

        Returns:
            AcquisitionFunction: Configured acquisition function.
        """
        if (
            "lb" in inspect.signature(self.acqf).parameters
            and "ub" in inspect.signature(self.acqf).parameters
        ):
            if self.acqf == AnalyticExpectedUtilityOfBestOption:
                return self.acqf(pref_model=model, lb=self.lb, ub=self.ub)

            self.lb = self.lb.to(model.device)
            self.ub = self.ub.to(model.device)
            if self.acqf in self.baseline_requiring_acqfs:
                return self.acqf(
                    model,
                    model.train_inputs[0],
                    lb=self.lb,
                    ub=self.ub,
                    **self.acqf_kwargs,
                )

            return self.acqf(model=model, lb=self.lb, ub=self.ub, **self.acqf_kwargs)

        if self.acqf == AnalyticExpectedUtilityOfBestOption:
            return self.acqf(pref_model=model)

        if self.acqf in self.baseline_requiring_acqfs:
            return self.acqf(model, model.train_inputs[0], **self.acqf_kwargs)

        return self.acqf(model=model, **self.acqf_kwargs)

    def gen(self, num_points: int, model: ModelProtocol, **gen_options) -> torch.Tensor:
        """Query next point(s) to run by optimizing the acquisition function.
        Args:
            num_points (int): Number of points to query.
            model (ModelProtocol): Fitted model of the data.
        Returns:
            torch.Tensor: Next set of point(s) to evaluate, [num_points x dim].
        """

        if self.stimuli_per_trial == 2:
            qbatch_points = self._gen(
                num_points=num_points * 2, model=model, **gen_options
            )

            # output of super() is (q, dim) but the contract is (num_points, dim, 2)
            # so we need to split q into q and pairs and then move the pair dim to the end
            return qbatch_points.reshape(num_points, 2, -1).swapaxes(-1, -2)

        else:
            return self._gen(num_points=num_points, model=model, **gen_options)

    def _gen(
        self, num_points: int, model: ModelProtocol, **gen_options: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Generates the next query points by optimizing the acquisition function.

        Args:
            num_points (int): Number of points to query.
            model (ModelProtocol): Fitted model of the data.
            gen_options (Dict[str, Any]): Additional options for generating points, such as custom configurations.

        Returns:
            torch.Tensor: Next set of points to evaluate, with shape [num_points x dim].
        """
        # eval should be inherited from superclass
        model.eval()  # type: ignore
        acqf = self._instantiate_acquisition_fn(model)

        if hasattr(model, "device"):
            self.lb = self.lb.to(model.device)
            self.ub = self.ub.to(model.device)

        logger.info("Starting gen...")
        starttime = time.time()

        new_candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=torch.stack([self.lb, self.ub]),
            q=num_points,
            num_restarts=self.restarts,
            raw_samples=self.samps,
            timeout_sec=self.max_gen_time,
            **gen_options,
        )

        logger.info(f"Gen done, time={time.time()-starttime}")
        return new_candidate

    @classmethod
    def from_config(cls, config: Config) -> "OptimizeAcqfGenerator":
        """
        Creates an instance of OptimizeAcqfGenerator from a configuration object.

        Args:
            config (Config): Configuration object containing initialization parameters.

        Returns:
            OptimizeAcqfGenerator: A configured instance of OptimizeAcqfGenerator with specified acquisition function,
            restart and sample parameters, maximum generation time, and stimuli per trial.
        """
        classname = cls.__name__
        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        acqf = config.getobj(classname, "acqf", fallback=None)
        extra_acqf_args = cls._get_acqf_options(acqf, config)
        stimuli_per_trial = config.getint(classname, "stimuli_per_trial")
        restarts = config.getint(classname, "restarts", fallback=10)
        samps = config.getint(classname, "samps", fallback=1000)
        max_gen_time = config.getfloat(classname, "max_gen_time", fallback=None)

        return cls(
            lb=lb,
            ub=ub,
            acqf=acqf,
            acqf_kwargs=extra_acqf_args,
            restarts=restarts,
            samps=samps,
            max_gen_time=max_gen_time,
            stimuli_per_trial=stimuli_per_trial,
        )
