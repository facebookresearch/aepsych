#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time
import warnings
from itertools import product
from typing import Any

import torch
from aepsych.acquisition.lookahead import LookaheadAcquisitionFunction
from aepsych.config import Config
from aepsych.generators.base import AcqfGenerator
from aepsych.models.base import AEPsychModelMixin
from aepsych.utils_logging import getLogger
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf, optimize_acqf_mixed

logger = getLogger()


class OptimizeAcqfGenerator(AcqfGenerator):
    """Generator that chooses points by minimizing an acquisition function."""

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        acqf: AcquisitionFunction,
        acqf_kwargs: dict[str, Any] | None = None,
        restarts: int = 10,
        samps: int = 1024,
        max_gen_time: float | None = None,
        stimuli_per_trial: int = 1,
    ) -> None:
        """Initialize OptimizeAcqfGenerator.
        Args:
            lb (torch.Tensor): Lower bounds for the optimization.
            ub (torch.Tensor): Upper bounds for the optimization.
            acqf (AcquisitionFunction): Acquisition function to use.
            acqf_kwargs (dict[str, object], optional): Extra arguments to
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
        model: AEPsychModelMixin,
        fixed_features: dict[int, float] | None = None,
        X_pending: torch.Tensor | None = None,
        **gen_options,
    ) -> torch.Tensor:
        """Query next point(s) to run by optimizing the acquisition function.
        Args:
            num_points (int): Number of points to query.
            model (AEPsychModelMixin): Fitted model of the data.
            fixed_features (dict[int, float], optional): The values where the specified
                parameters should be at when generating. Should be a dictionary where
                the keys are the indices of the parameters to fix and the values are the
                values to fix them at.
            X_pending (torch.Tensor, option): Points that have been generated but not
                evaluated yet. For sequential generation.
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

        if hasattr(acqf, "set_X_pending"):
            acqf.set_X_pending(X_pending)

        if isinstance(acqf, LookaheadAcquisitionFunction) and num_points > 1:
            warnings.warn(
                f"{num_points} points were requested, but `{acqf.__class__.__name__}` can only generate one point at a time, returning only 1.",
                stacklevel=2,
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
        model: AEPsychModelMixin,
        acqf: AcquisitionFunction,
        fixed_features: dict[int, float] | None = None,
        **gen_options: dict[str, Any],
    ) -> torch.Tensor:
        """
        Generates the next query points by optimizing the acquisition function.

        Args:
            num_points (int): Number of points to query.
            model (AEPsychModelMixin): Fitted model of the data.
            acqf (AcquisitionFunction): Acquisition function.
            fixed_features (dict[int, float], optional): The values where the specified
                parameters should be at when generating. Should be a dictionary where
                the keys are the indices of the parameters to fix and the values are the
                values to fix them at.
            gen_options (dict[str, Any]): Additional options for generating points, such as custom configurations.

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


class MixedOptimizeAcqfGenerator(OptimizeAcqfGenerator):
    """A variant of OptimizeAcqfGenerator that supports mixed parameter types
    (namely continuous and categorical parameters)."""

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        categorical_parameters: dict[int, int],
        acqf: AcquisitionFunction,
        acqf_kwargs: dict[str, Any] | None = None,
        restarts: int = 10,
        samps: int = 1024,
        max_gen_time: float | None = None,
        stimuli_per_trial: int = 1,
    ) -> None:
        """Initialize OptimizeAcqfGenerator.
        Args:
            lb (torch.Tensor): Lower bounds for the optimization.
            ub (torch.Tensor): Upper bounds for the optimization.
            categorical_parameters (dict[int, int]): A dictionary mapping the indices of the categorical
                parameters to the number of categories.
            acqf (AcquisitionFunction): Acquisition function to use.
            acqf_kwargs (dict[str, object], optional): Extra arguments to
                pass to acquisition function. Defaults to no arguments.
            restarts (int): Number of restarts for acquisition function optimization. Defaults to 10.
            samps (int): Number of samples for quasi-random initialization of the acquisition function optimizer. Defaults to 1000.
            max_gen_time (float, optional): Maximum time (in seconds) to optimize the acquisition function. Defaults to None.
            stimuli_per_trial (int): Number of stimuli per trial. Defaults to 1.
        """
        super().__init__(
            lb=lb,
            ub=ub,
            acqf=acqf,
            acqf_kwargs=acqf_kwargs,
            restarts=restarts,
            samps=samps,
            max_gen_time=max_gen_time,
            stimuli_per_trial=stimuli_per_trial,
        )

        # Make every possible combination of categorical values in a list
        cat_indices = list(categorical_parameters.keys())
        cat_values = [range(n) for n in categorical_parameters.values()]
        categorical_combos = []
        for combo in product(*cat_values):
            # Unpack combo into a dictionary
            categorical_combos.append(dict(zip(cat_indices, [float(x) for x in combo])))

        self.categorical_combos = categorical_combos

    def _gen(
        self,
        num_points: int,
        model: AEPsychModelMixin,
        acqf: AcquisitionFunction,
        fixed_features: dict[int, float] | None = None,
        **gen_options: dict[str, Any],
    ) -> torch.Tensor:
        """
        Generates the next query points by optimizing the acquisition function.

        Args:
            num_points (int): Number of points to query.
            model (AEPsychModelMixin): Fitted model of the data.
            acqf (AcquisitionFunction): Acquisition function.
            fixed_features (dict[int, float], optional): The values where the specified
                parameters should be at when generating. Should be a dictionary where
                the keys are the indices of the parameters to fix and the values are the
                values to fix them at.
            gen_options (dict[str, Any]): Additional options for generating points, such as custom configurations.

        Returns:
            torch.Tensor: Next set of points to evaluate, with shape [num_points x dim].
        """
        if fixed_features is not None and len(fixed_features) > 0:
            raise NotImplementedError(
                "Fixed features are not supported for mixed parameter types."
            )
        logger.info("Starting gen...")
        starttime = time.time()

        new_candidate, _ = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=torch.stack([self.lb, self.ub]),
            q=num_points,
            fixed_features_list=self.categorical_combos,
            num_restarts=self.restarts,
            raw_samples=self.samps,
            timeout_sec=self.max_gen_time,
            **gen_options,
        )

        logger.info(f"Gen done, time={time.time() - starttime}")
        return new_candidate

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = super().get_config_options(config, name, options)

        # Figure out discrete parameters
        par_names = config.getlist("common", "parnames", element_type=str)
        discrete_params = {}
        for i, par_name in enumerate(par_names):
            if config.get(par_name, "par_type") == "categorical":
                discrete_params[i] = len(
                    config.getlist(par_name, "choices", element_type=str)
                )

        if len(discrete_params) == 0:
            raise ValueError("No categorical parameters found")

        options["categorical_parameters"] = discrete_params

        return options
