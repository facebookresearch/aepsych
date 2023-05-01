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
from aepsych.models.surrogate import AEPsychSurrogate
from aepsych.utils_logging import getLogger
from ax.modelbridge import Models
from ax.modelbridge.registry import Cont_X_trans
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf
from botorch.utils import draw_sobol_samples

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
                This is only loosely followed by scipy's optimizer, so consider using a number about 1/3 or
                less of what your true upper bound is.
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

        if self.max_gen_time is None:
            new_candidate, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor(np.c_[model.lb, model.ub]).T.to(train_x),
                q=num_points,
                num_restarts=self.restarts,
                raw_samples=self.samps,
                **gen_options,
            )
        else:
            # figure out how long evaluating a single samp
            starttime = time.time()
            _ = acqf(train_x[0:num_points, :])
            single_eval_time = time.time() - starttime

            # only a heuristic for total num evals since everything is stochastic,
            # but the reasoning is: we initialize with self.samps samps, subsample
            # self.restarts from them in proportion to the value of the acqf, and
            # run that many optimization. So:
            # total_time = single_eval_time * n_eval * restarts + single_eval_time * samps
            # and we solve for n_eval
            n_eval = int(
                (self.max_gen_time - single_eval_time * self.samps)
                / (single_eval_time * self.restarts)
            )
            if n_eval > 10:
                # heuristic, if we can't afford 10 evals per restart, just use quasi-random search
                options = {"maxfun": n_eval}
                logger.info(f"gen maxfun is {n_eval}")

                new_candidate, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=torch.tensor(np.c_[model.lb, model.ub]).T.to(train_x),
                    q=num_points,
                    num_restarts=self.restarts,
                    raw_samples=self.samps,
                    options=options,
                )
            else:
                logger.info(f"gen maxfun is {n_eval}, falling back to random search...")
                nsamp = max(int(self.max_gen_time / single_eval_time), 10)
                # Generate the points at which to sample
                bounds = torch.stack((model.lb, model.ub))

                X = draw_sobol_samples(bounds=bounds, n=nsamp, q=num_points)

                acqvals = acqf(X)

                best_indx = torch.argmax(acqvals, dim=0)
                new_candidate = X[best_indx]

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

        max_fit_time = model_options["max_fit_time"]

        model_kwargs = {
            "surrogate": AEPsychSurrogate(
                botorch_model_class=model_class,
                mll_class=model_class.get_mll_class(),
                model_options=model_options,
                max_fit_time=max_fit_time,
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
        restarts = config.getint(classname, "restarts", fallback=10)
        samps = config.getint(classname, "samps", fallback=1000)
        return {"restarts": restarts, "samps": samps}
