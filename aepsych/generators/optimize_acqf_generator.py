#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Optional

import torch
from aepsych.acquisition import MCLevelSetEstimation
from aepsych.acquisition.lookahead import LookaheadAcquisitionFunction
from aepsych.config import Config
from aepsych.generators.base import AcqfGenerator
from aepsych.models.base import ModelProtocol
from aepsych.utils_logging import getLogger
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf

logger = getLogger()


class OptimizeAcqfGenerator(AcqfGenerator):
    """Generator that chooses points by minimizing an acquisition function."""

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        acqf: AcquisitionFunction,
        acqf_kwargs: Optional[Dict[str, Any]] = None,
        restarts: int = 10,
        samps: int = 1024,
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
        super().__init__(acqf=acqf, acqf_kwargs=acqf_kwargs)
        self.restarts = restarts
        self.samps = samps
        self.max_gen_time = max_gen_time
        self.stimuli_per_trial = stimuli_per_trial
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

    def gen(
        self,
        num_points: int,
        model: ModelProtocol,
        fixed_features: Optional[Dict[int, float]] = None,
        **gen_options,
    ) -> torch.Tensor:
        """Query next point(s) to run by optimizing the acquisition function.
        Args:
            num_points (int): Number of points to query.
            model (ModelProtocol): Fitted model of the data.
            fixed_features (Dict[int, float], optional): The values where the specified
                parameters should be at when generating. Should be a dictionary where
                the keys are the indices of the parameters to fix and the values are the
                values to fix them at.
            **gen_options: Additional options for generating points, such as custom configurations.

        Returns:
            torch.Tensor: Next set of point(s) to evaluate, [num_points x dim].
        """
        # eval should be inherited from superclass
        model.eval()  # type: ignore
        if hasattr(model, "device"):
            self.lb = self.lb.to(model.device)
            self.ub = self.ub.to(model.device)

        acqf = self._instantiate_acquisition_fn(model)

        if isinstance(acqf, LookaheadAcquisitionFunction) and num_points > 1:
            warnings.warn(
                f"{num_points} points were requested, but `{acqf.__class__.__name__}` can only generate one point at a time, returning only 1."
            )
            num_points = 1

        if self.stimuli_per_trial == 2:
            qbatch_points = self._gen(
                num_points=num_points * 2,
                model=model,
                acqf=acqf,
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
                acqf=acqf,
                fixed_features=fixed_features,
                **gen_options,
            )

    def _gen(
        self,
        num_points: int,
        model: ModelProtocol,
        acqf: AcquisitionFunction,
        fixed_features: Optional[Dict[int, float]] = None,
        **gen_options: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Generates the next query points by optimizing the acquisition function.

        Args:
            num_points (int): Number of points to query.
            model (ModelProtocol): Fitted model of the data.
            acqf (AcquisitionFunction): Acquisition function.
            fixed_features (Dict[int, float], optional): The values where the specified
                parameters should be at when generating. Should be a dictionary where
                the keys are the indices of the parameters to fix and the values are the
                values to fix them at.
            gen_options (Dict[str, Any]): Additional options for generating points, such as custom configurations.

        Returns:
            torch.Tensor: Next set of points to evaluate, with shape [num_points x dim].
        """
        logger.info("Starting gen...")
        starttime = time.time()

        new_candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=torch.stack([self.lb, self.ub]),
            q=num_points,
            num_restarts=self.restarts,
            raw_samples=self.samps,
            timeout_sec=self.max_gen_time,
            fixed_features=fixed_features,
            **gen_options,
        )

        logger.info(f"Gen done, time={time.time() - starttime}")
        return new_candidate

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get configuration options for the generator.

        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the generator, defaults to None.
            options (Dict[str, Any], optional): Additional options, defaults to None.

        Returns:
            Dict[str, Any]: Configuration options for the generator.
        """
        options = options or {}
        options.update(super().get_config_options(config, name, options))

        classname = name or cls.__name__
        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        stimuli_per_trial = config.getint(classname, "stimuli_per_trial")
        restarts = config.getint(classname, "restarts", fallback=10)
        samps = config.getint(classname, "samps", fallback=1024)
        max_gen_time = config.getfloat(classname, "max_gen_time", fallback=None)

        options.update(
            {
                "lb": lb,
                "ub": ub,
                "restarts": restarts,
                "samps": samps,
                "max_gen_time": max_gen_time,
                "stimuli_per_trial": stimuli_per_trial,
            }
        )

        return options
