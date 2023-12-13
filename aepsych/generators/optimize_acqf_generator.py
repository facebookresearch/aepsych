#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time
from inspect import signature
from typing import Any, cast, Dict, Optional

import numpy as np
import torch
from aepsych.acquisition.acquisition import AEPsychAcquisition
from aepsych.config import Config, ConfigurableMixin
from aepsych.generators.base import AEPsychGenerationStep, AEPsychGenerator
from aepsych.models.base import ModelProtocol
from ax.models.torch.botorch_modular.surrogate import Surrogate
from aepsych.utils_logging import getLogger
from ax.modelbridge import Models
from ax.modelbridge.registry import Cont_X_trans
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf

logger = getLogger()


class OptimizeAcqfGenerator(AEPsychGenerator):
    """Generator that chooses points by minimizing an acquisition function."""

    def __init__(
        self,
        acqf: AcquisitionFunction,
        acqf_kwargs: Optional[Dict[str, Any]] = None,
        restarts: int = 10,
        samps: int = 1000,
        max_gen_time: Optional[float] = None,
        stimuli_per_trial: int = 1,
    ) -> None:
        """Initialize OptimizeAcqfGenerator.
        Args:
            acqf (AcquisitionFunction): Acquisition function to use.
            acqf_kwargs (Dict[str, object], optional): Extra arguments to
                pass to acquisition function. Defaults to no arguments.
            restarts (int): Number of restarts for acquisition function optimization.
            samps (int): Number of samples for quasi-random initialization of the acquisition function optimizer.
            max_gen_time (optional, float): Maximum time (in seconds) to optimize the acquisition function.
        """

        if acqf_kwargs is None:
            acqf_kwargs = {}
        self.acqf = acqf
        self.acqf_kwargs = acqf_kwargs
        self.restarts = restarts
        self.samps = samps
        self.max_gen_time = max_gen_time
        self.stimuli_per_trial = stimuli_per_trial

    def _instantiate_acquisition_fn(self, model: ModelProtocol):
        if self.acqf == AnalyticExpectedUtilityOfBestOption:
            return self.acqf(pref_model=model)

        if self.acqf in self.baseline_requiring_acqfs:
            return self.acqf(
                model=model, X_baseline=model.train_inputs[0], **self.acqf_kwargs
            )
        else:
            return self.acqf(model=model, **self.acqf_kwargs)

    def gen(self, num_points: int, model: ModelProtocol, **gen_options) -> torch.Tensor:
        """Query next point(s) to run by optimizing the acquisition function.
        Args:
            num_points (int, optional): Number of points to query.
            model (ModelProtocol): Fitted model of the data.
        Returns:
            np.ndarray: Next set of point(s) to evaluate, [num_points x dim].
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
        self, num_points: int, model: ModelProtocol, **gen_options
    ) -> torch.Tensor:
        # eval should be inherited from superclass
        model.eval()  # type: ignore
        train_x = model.train_inputs[0]
        acqf = self._instantiate_acquisition_fn(model)

        logger.info("Starting gen...")
        starttime = time.time()

        new_candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=torch.tensor(np.c_[model.lb, model.ub]).T.to(train_x),
            q=num_points,
            num_restarts=self.restarts,
            raw_samples=self.samps,
            timeout_sec=self.max_gen_time,
            **gen_options,
        )

        logger.info(f"Gen done, time={time.time()-starttime}")
        return new_candidate

    @classmethod
    def from_config(cls, config: Config):
        classname = cls.__name__
        acqf = config.getobj(classname, "acqf", fallback=None)
        extra_acqf_args = cls._get_acqf_options(acqf, config)
        stimuli_per_trial = config.getint(classname, "stimuli_per_trial")
        restarts = config.getint(classname, "restarts", fallback=10)
        samps = config.getint(classname, "samps", fallback=1000)
        max_gen_time = config.getfloat(classname, "max_gen_time", fallback=None)

        return cls(
            acqf=acqf,
            acqf_kwargs=extra_acqf_args,
            restarts=restarts,
            samps=samps,
            max_gen_time=max_gen_time,
            stimuli_per_trial=stimuli_per_trial,
        )

    @classmethod
    def get_config_options(cls, config: Config, name: str):
        return AxOptimizeAcqfGenerator.get_config_options(config, name)


class AxOptimizeAcqfGenerator(AEPsychGenerationStep, ConfigurableMixin):
    @classmethod
    def get_config_options(cls, config: Config, name: str) -> Dict:
        classname = "OptimizeAcqfGenerator"

        model_class = config.getobj(name, "model", fallback=None)
        model_options = model_class.get_config_options(config)

        acqf_cls = config.getobj(name, "acqf", fallback=None)
        if acqf_cls is None:
            acqf_cls = config.getobj(classname, "acqf")

        acqf_options = cls._get_acqf_options(acqf_cls, config)
        gen_options = cls._get_gen_options(config)

        model_kwargs = {
            "surrogate": Surrogate(
                botorch_model_class=model_class,
                mll_class=model_class.get_mll_class(),
                model_options=model_options,
            ),
            "acquisition_class": AEPsychAcquisition,
            "botorch_acqf_class": acqf_cls,
            "acquisition_options": acqf_options,
            # The Y transforms are removed because they are incompatible with our thresholding-finding acqfs
            # The target value doesn't get transformed, so it searches for the target in the wrong space.
            "transforms": Cont_X_trans,  # TODO: Make LSE acqfs compatible with Y transforms
        }

        opts = {
            "model": Models.BOTORCH_MODULAR,
            "model_kwargs": model_kwargs,
            "model_gen_kwargs": gen_options,
        }

        opts.update(super().get_config_options(config, name))

        return opts

    @classmethod
    def _get_acqf_options(cls, acqf: AcquisitionFunction, config: Config):
        class MissingValue:
            pass

        if acqf is not None:
            acqf_name = acqf.__name__

            acqf_args_expected = signature(acqf).parameters.keys()
            acqf_args = {
                k: config.getobj(
                    acqf_name,
                    k,
                    fallback_type=float,
                    fallback=MissingValue(),
                    warn=False,
                )
                for k in acqf_args_expected
            }
            acqf_args = {
                k: v for k, v in acqf_args.items() if not isinstance(v, MissingValue)
            }
            for k, v in acqf_args.items():
                if hasattr(v, "from_config"):  # configure if needed
                    acqf_args[k] = cast(Any, v).from_config(config)
                elif isinstance(v, type):  # instaniate a class if needed
                    acqf_args[k] = v()
        else:
            acqf_args = {}

        return acqf_args

    @classmethod
    def _get_gen_options(cls, config: Config):
        classname = "OptimizeAcqfGenerator"
        restarts = config.getint(classname, "num_restarts", fallback=10)
        samps = config.getint(classname, "raw_samples", fallback=1024)
        timeout_sec = config.getfloat(classname, "max_gen_time", fallback=None)
        optimizer_kwargs = {
            "optimizer_kwargs": {
                "num_restarts": restarts,
                "raw_samples": samps,
                "timeout_sec": timeout_sec,
            }
        }
        return {"model_gen_options": optimizer_kwargs}
