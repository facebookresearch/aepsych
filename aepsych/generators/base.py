#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc
from inspect import signature
from typing import Any, Dict, Generic, Protocol, TypeVar, runtime_checkable, Optional
import copy

import numpy as np
from aepsych.config import Config, ConfigurableMixin
from aepsych.models.base import AEPsychMixin
from ax.core.experiment import Experiment
from ax.modelbridge.generation_node import GenerationStep
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.core.search_space import SearchSpace
from ax.core.optimization_config import OptimizationConfig
from ax.core.data import Data
from ax.utils.common.constants import Keys
from ax.modelbridge.registry import _extract_model_state_after_gen, ModelRegistryBase
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.common.logger import _round_floats_for_logging, get_logger
from botorch.acquisition import (
    AcquisitionFunction,
    NoisyExpectedImprovement,
    qNoisyExpectedImprovement,
)
from logging import Logger

from .completion_criterion import completion_criteria

logger: Logger = get_logger(__name__)

AEPsychModelType = TypeVar("AEPsychModelType", bound=AEPsychMixin)


@runtime_checkable
class AcqArgProtocol(Protocol):
    @classmethod
    def from_config(cls, config: Config) -> Any:
        pass


class AEPsychGenerator(abc.ABC, Generic[AEPsychModelType]):
    """Abstract base class for generators, which are responsible for generating new points to sample."""

    _requires_model = True
    baseline_requiring_acqfs = [qNoisyExpectedImprovement, NoisyExpectedImprovement]
    stimuli_per_trial = 1

    def __init__(
        self,
    ) -> None:
        pass

    @abc.abstractmethod
    def gen(self, num_points: int, model: AEPsychModelType) -> np.ndarray:
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Config):
        pass

    @classmethod
    def _get_acqf_options(cls, acqf: AcquisitionFunction, config: Config):
        if acqf is not None:
            acqf_name = acqf.__name__

            # model is not an extra arg, it's a default arg
            acqf_args_expected = [
                i for i in list(signature(acqf).parameters.keys()) if i != "model"
            ]

            # this is still very ugly
            extra_acqf_args = {}
            if acqf_name in config:
                full_section = config[acqf_name]
                for k in acqf_args_expected:
                    # if this thing is configured
                    if k in full_section.keys():
                        # if it's an object make it an object
                        if full_section[k] in Config.registered_names.keys():
                            extra_acqf_args[k] = config.getobj(acqf_name, k)
                        else:
                            # otherwise try a float
                            try:
                                extra_acqf_args[k] = config.getfloat(acqf_name, k)
                            # finally just return a string
                            except ValueError:
                                extra_acqf_args[k] = config.get(acqf_name, k)

            # next, do more processing
            for k, v in extra_acqf_args.items():
                if hasattr(v, "from_config"):  # configure if needed
                    assert isinstance(v, AcqArgProtocol)  # make mypy happy
                    extra_acqf_args[k] = v.from_config(config)
                elif isinstance(v, type):  # instaniate a class if needed
                    extra_acqf_args[k] = v()
        else:
            extra_acqf_args = {}

        return extra_acqf_args


class AEPsychGenerationStep(GenerationStep, ConfigurableMixin, abc.ABC):
    def __init__(self, name, **kwargs):
        self.refit_every = kwargs.get("refit_every", 1)
        if "refit_every" in kwargs:
            del kwargs["refit_every"]
        super().__init__(num_trials=-1, **kwargs)
        self.name = name

    @classmethod
    def get_config_options(cls, config: Config, name: str) -> Dict:
        criteria = []
        for crit in completion_criteria:
            # TODO: Figure out how to convince mypy that CompletionCriterion have `from_config`
            criterion = crit.from_config(config, name)  # type: ignore
            criteria.append(criterion)
        options = {"completion_criteria": criteria, "name": name}
        return options

    def finished(self, experiment: Experiment):
        finished = all(
            [criterion.is_met(experiment) for criterion in self.completion_criteria]
        )
        return finished

    def fit(
        self,
        experiment: Experiment,
        data: Data,
        num_trials: int,
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Fits the specified models to the given experiment + data using
        the model kwargs set on each corresponding model spec and the kwargs
        passed to this method.

        NOTE: Local kwargs take precedence over the ones stored in
        ``ModelSpec.model_kwargs``.
        """
        self._model_spec_to_gen_from = None
        for model_spec in self.model_specs:
            tmp_kwargs = copy.deepcopy(kwargs)
            if (
                model_spec._fitted_model
                and model_spec.model_enum._name_ == "BOTORCH_MODULAR"
            ):
                tmp_kwargs.update(
                    {
                        "refit": not (num_trials % self.refit_every),
                        "state_dict": model_spec._fitted_model.model.surrogates[
                            Keys.ONLY_SURROGATE
                        ].model.state_dict(),
                    }
                )

            model_spec.fit(  # Stores the fitted model as `model_spec._fitted_model`
                experiment=experiment,
                data=data,
                search_space=search_space,
                optimization_config=optimization_config,
                **tmp_kwargs,
            )


class AEPsychGenerationStrategy(GenerationStrategy):
    def _fit_current_model(self, data: Data) -> None:
        """Instantiate the current model with all available data."""
        # If last generator run's index matches the current step, extract
        # model state from last generator run and pass it to the model
        # being instantiated in this function.
        lgr = self.last_generator_run
        # NOTE: This will not be easily compatible with `GenerationNode`;
        # will likely need to find last generator run per model. Not a problem
        # for now though as GS only allows `GenerationStep`-s for now.
        # Potential solution: store generator runs on `GenerationStep`-s and
        # split them per-model there.
        model_state_on_lgr = {}
        model_on_curr = self._curr.model
        if (
            lgr is not None
            and lgr._generation_step_index == self._curr.index
            and lgr._model_state_after_gen
        ):
            if self.model or isinstance(model_on_curr, ModelRegistryBase):
                # TODO[drfreund]: Consider moving this to `GenerationStep` or
                # `GenerationNode`.
                model_cls = (
                    self.model.model.__class__
                    if self.model is not None
                    # NOTE: This checked cast is save per the OR-statement in last line
                    # of the IF-check above.
                    else checked_cast(ModelRegistryBase, model_on_curr).model_class
                )
                model_state_on_lgr = _extract_model_state_after_gen(
                    generator_run=lgr,
                    model_class=model_cls,
                )
            else:
                logger.warning(  # pragma: no cover
                    "While model state after last call to `gen` was recorded on the "
                    "las generator run produced by this generation strategy, it could"
                    " not be applied because model for this generation step is defined"
                    f" via factory function: {self._curr.model}. Generation strategies"
                    " with factory functions do not support reloading from a stored "
                    "state."
                )

        if not data.df.empty:
            trial_indices_in_data = sorted(data.df["trial_index"].unique())
            logger.debug(f"Fitting model with data for trials: {trial_indices_in_data}")
        self._curr.fit(
            experiment=self.experiment,
            data=data,
            num_trials=self.experiment.num_asks,
            **model_state_on_lgr,
        )
        self._model = self._curr.model_spec.fitted_model
