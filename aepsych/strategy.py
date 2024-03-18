#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import time
import warnings

from copy import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch

from aepsych.config import Config, ConfigurableMixin
from aepsych.generators.base import AEPsychGenerationStep, AEPsychGenerator
from aepsych.generators.sobol_generator import AxSobolGenerator, SobolGenerator
from aepsych.models.base import ModelProtocol
from aepsych.utils import (
    _process_bounds,
    get_bounds,
    get_objectives,
    get_parameters,
    make_scaled_sobol,
)
from aepsych.utils_logging import getLogger
from ax.core.base_trial import TrialStatus
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.plot.contour import interact_contour
from ax.plot.slice import plot_slice
from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import render
from botorch.exceptions.errors import ModelFittingError

logger = getLogger()


def ensure_model_is_fresh(f):
    def wrapper(self, *args, **kwargs):
        if self.can_fit and not self._model_is_fresh:
            starttime = time.time()
            if self._count % self.refit_every == 0 or self.refit_every == 1:
                logger.info("Starting fitting (no warm start)...")
                # don't warm start
                self.fit()
            else:
                logger.info("Starting fitting (warm start)...")
                # warm start
                self.update()
            logger.info(f"Fitting done, took {time.time()-starttime}")
        self._model_is_fresh = True
        return f(self, *args, **kwargs)

    return wrapper


class Strategy(object):
    """Object that combines models and generators to generate points to sample."""

    _n_eval_points: int = 1000

    def __init__(
        self,
        generator: AEPsychGenerator,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        stimuli_per_trial: int,
        outcome_types: Sequence[Type[str]],
        dim: Optional[int] = None,
        min_total_tells: int = 0,
        min_asks: int = 0,
        model: Optional[ModelProtocol] = None,
        refit_every: int = 1,
        min_total_outcome_occurrences: int = 1,
        max_asks: Optional[int] = None,
        keep_most_recent: Optional[int] = None,
        min_post_range: Optional[float] = None,
        name: str = "",
        run_indefinitely: bool = False,
    ):
        """Initialize the strategy object.

        Args:
            generator (AEPsychGenerator): The generator object that determines how points are sampled.
            lb (Union[numpy.ndarray, torch.Tensor]): Lower bounds of the parameters.
            ub (Union[numpy.ndarray, torch.Tensor]): Upper bounds of the parameters.
            dim (int, optional): The number of dimensions in the parameter space. If None, it is inferred from the size
                of lb and ub.
            min_total_tells (int): The minimum number of total observations needed to complete this strategy.
            min_asks (int): The minimum number of points that should be generated from this strategy.
            model (ModelProtocol, optional): The AEPsych model of the data.
            refit_every (int): How often to refit the model from scratch.
            min_total_outcome_occurrences (int): The minimum number of total observations needed for each outcome before the strategy will finish.
                Defaults to 1 (i.e., for binary outcomes, there must be at least one "yes" trial and one "no" trial).
            max_asks (int, optional): The maximum number of trials to generate using this strategy.
                If None, there is no upper bound (default).
            keep_most_recent (int, optional): Experimental. The number of most recent data points that the model will be fitted on.
                This may be useful for discarding noisy data from trials early in the experiment that are not as informative
                as data collected from later trials. When None, the model is fitted on all data.
            min_post_range (float, optional): Experimental. The required difference between the posterior's minimum and maximum value in
                probablity space before the strategy will finish. Ignored if None (default).
            name (str): The name of the strategy. Defaults to the empty string.
            run_indefinitely (bool): If true, the strategy will run indefinitely until finish() is explicitly called. Other stopping criteria will
                be ignored. Defaults to False.
        """
        self.is_finished = False

        if run_indefinitely:
            warnings.warn(
                f"Strategy {name} will run indefinitely until finish() is explicitly called. Other stopping criteria will be ignored."
            )

        elif min_total_tells > 0 and min_asks > 0:
            warnings.warn(
                "Specifying both min_total_tells and min_asks > 0 may lead to unintended behavior."
            )

        if model is not None:
            assert (
                len(outcome_types) == model._num_outputs
            ), f"Strategy has {len(outcome_types)} outcomes, but model {type(model).__name__} supports {model._num_outputs}!"
            assert (
                stimuli_per_trial == model.stimuli_per_trial
            ), f"Strategy has {stimuli_per_trial} stimuli_per_trial, but model {type(model).__name__} supports {model.stimuli_per_trial}!"

            if isinstance(model.outcome_type, str):
                assert (
                    len(outcome_types) == 1 and outcome_types[0] == model.outcome_type
                ), f"Strategy outcome types is {outcome_types} but model outcome type is {model.outcome_type}!"
            else:
                assert set(outcome_types) == set(
                    model.outcome_type
                ), f"Strategy outcome types is {outcome_types} but model outcome type is {model.outcome_type}!"

        self.run_indefinitely = run_indefinitely
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        self.min_total_outcome_occurrences = min_total_outcome_occurrences
        self.max_asks = max_asks or generator.max_asks
        self.keep_most_recent = keep_most_recent

        self.min_post_range = min_post_range
        if self.min_post_range is not None:
            assert model is not None, "min_post_range must be None if model is None!"
            self.eval_grid = make_scaled_sobol(
                lb=self.lb, ub=self.ub, size=self._n_eval_points
            )

        self.x = None
        self.y = None
        self.n = 0
        self.min_asks = min_asks
        self._count = 0
        self.min_total_tells = min_total_tells
        self.stimuli_per_trial = stimuli_per_trial
        self.outcome_types = outcome_types

        if self.stimuli_per_trial == 1:
            self.event_shape: Tuple[int, ...] = (self.dim,)

        if self.stimuli_per_trial == 2:
            self.event_shape = (self.dim, self.stimuli_per_trial)

        self.model = model
        self.refit_every = refit_every
        self._model_is_fresh = False
        self.generator = generator
        self.has_model = self.model is not None
        if self.generator._requires_model:
            assert self.model is not None, f"{self.generator} requires a model!"

        if self.min_asks == self.min_total_tells == 0:
            warnings.warn(
                "strategy.min_asks == strategy.min_total_tells == 0. This strategy will not generate any points!",
                UserWarning,
            )

        self.name = name

    def normalize_inputs(self, x, y):
        """converts inputs into normalized format for this strategy

        Args:
            x (np.ndarray): training inputs
            y (np.ndarray): training outputs

        Returns:
            x (np.ndarray): training inputs, normalized
            y (np.ndarray): training outputs, normalized
            n (int): number of observations
        """
        assert (
            x.shape == self.event_shape or x.shape[1:] == self.event_shape
        ), f"x shape should be {self.event_shape} or batch x {self.event_shape}, instead got {x.shape}"

        if x.shape == self.event_shape:
            x = x[None, :]

        if self.x is None:
            x = np.r_[x]
        else:
            x = np.r_[self.x, x]

        if self.y is None:
            y = np.r_[y]
        else:
            y = np.r_[self.y, y]

        n = y.shape[0]

        return torch.Tensor(x), torch.Tensor(y), n

    # TODO: allow user to pass in generator options
    @ensure_model_is_fresh
    def gen(self, num_points: int = 1):
        """Query next point(s) to run by optimizing the acquisition function.

        Args:
            num_points (int, optional): Number of points to query. Defaults to 1.
            Other arguments are forwared to underlying model.

        Returns:
            np.ndarray: Next set of point(s) to evaluate, [num_points x dim].
        """
        self._count = self._count + num_points
        return self.generator.gen(num_points, self.model)

    @ensure_model_is_fresh
    def get_max(self, constraints=None, probability_space=False, max_time=None):
        constraints = constraints or {}
        return self.model.get_max(
            constraints, probability_space=probability_space, max_time=max_time
        )

    @ensure_model_is_fresh
    def get_min(self, constraints=None, probability_space=False, max_time=None):
        constraints = constraints or {}
        return self.model.get_min(
            constraints, probability_space=probability_space, max_time=max_time
        )

    @ensure_model_is_fresh
    def inv_query(self, y, constraints=None, probability_space=False, max_time=None):
        constraints = constraints or {}
        return self.model.inv_query(
            y, constraints, probability_space, max_time=max_time
        )

    @ensure_model_is_fresh
    def predict(self, x, probability_space=False):
        return self.model.predict(x=x, probability_space=probability_space)

    @ensure_model_is_fresh
    def get_jnd(self, *args, **kwargs):
        return self.model.get_jnd(*args, **kwargs)

    @ensure_model_is_fresh
    def sample(self, x, num_samples=None):
        return self.model.sample(x, num_samples=num_samples)

    def finish(self):
        self.is_finished = True

    @property
    def finished(self):
        if self.is_finished:
            return True

        if self.run_indefinitely:
            return False

        if hasattr(self.generator, "finished"):  # defer to generator if possible
            return self.generator.finished

        if self.y is None:  # always need some data before switching strats
            return False

        if self.max_asks is not None and self._count >= self.max_asks:
            return True

        if "binary" in self.outcome_types:
            n_yes_trials = (self.y == 1).sum()
            n_no_trials = (self.y == 0).sum()
            sufficient_outcomes = (
                n_yes_trials >= self.min_total_outcome_occurrences
                and n_no_trials >= self.min_total_outcome_occurrences
            )
        else:
            sufficient_outcomes = True

        if self.min_post_range is not None:
            fmean, _ = self.model.predict(self.eval_grid, probability_space=True)
            meets_post_range = (fmean.max() - fmean.min()) >= self.min_post_range
        else:
            meets_post_range = True
        finished = (
            self._count >= self.min_asks
            and self.n >= self.min_total_tells
            and sufficient_outcomes
            and meets_post_range
        )
        return finished

    @property
    def can_fit(self):
        return self.has_model and self.x is not None and self.y is not None

    @property
    def n_trials(self):
        warnings.warn(
            "'n_trials' is deprecated and will be removed in a future release. Specify 'min_asks' instead.",
            DeprecationWarning,
        )
        return self.min_asks

    def add_data(self, x, y):
        self.x, self.y, self.n = self.normalize_inputs(x, y)
        self._model_is_fresh = False

    def fit(self):
        if self.can_fit:
            if self.keep_most_recent is not None:
                try:
                    self.model.fit(
                        self.x[-self.keep_most_recent :],
                        self.y[-self.keep_most_recent :],
                    )
                except ModelFittingError:
                    logger.warning(
                        "Failed to fit model! Predictions may not be accurate!"
                    )
            else:
                try:
                    self.model.fit(self.x, self.y)
                except ModelFittingError:
                    logger.warning(
                        "Failed to fit model! Predictions may not be accurate!"
                    )
        else:
            warnings.warn("Cannot fit: no model has been initialized!", RuntimeWarning)

    def update(self):
        if self.can_fit:
            if self.keep_most_recent is not None:
                try:
                    self.model.update(
                        self.x[-self.keep_most_recent :],
                        self.y[-self.keep_most_recent :],
                    )
                except ModelFittingError:
                    logger.warning(
                        "Failed to fit model! Predictions may not be accurate!"
                    )
            else:
                try:
                    self.model.update(self.x, self.y)
                except ModelFittingError:
                    logger.warning(
                        "Failed to fit model! Predictions may not be accurate!"
                    )
        else:
            warnings.warn("Cannot fit: no model has been initialized!", RuntimeWarning)

    @classmethod
    def from_config(cls, config: Config, name: str):
        lb = config.gettensor(name, "lb")
        ub = config.gettensor(name, "ub")
        dim = config.getint(name, "dim", fallback=None)

        stimuli_per_trial = config.getint(name, "stimuli_per_trial", fallback=1)
        outcome_types = config.getlist(name, "outcome_types", element_type=str)

        gen_cls = config.getobj(name, "generator", fallback=SobolGenerator)
        generator = gen_cls.from_config(config)

        model_cls = config.getobj(name, "model", fallback=None)
        if model_cls is not None:
            model = model_cls.from_config(config)
        else:
            model = None

        acqf_cls = config.getobj(name, "acqf", fallback=None)
        if acqf_cls is not None and hasattr(generator, "acqf"):
            if generator.acqf is None:
                generator.acqf = acqf_cls
                generator.acqf_kwargs = generator._get_acqf_options(acqf_cls, config)

        min_asks = config.getint(name, "min_asks", fallback=0)
        min_total_tells = config.getint(name, "min_total_tells", fallback=0)

        refit_every = config.getint(name, "refit_every", fallback=1)

        if model is not None and not generator._requires_model:
            if refit_every < min_asks:
                warnings.warn(
                    f"Strategy '{name}' has refit_every < min_asks even though its generator does not require a model. Consider making refit_every = min_asks to speed up point generation.",
                    UserWarning,
                )
        keep_most_recent = config.getint(name, "keep_most_recent", fallback=None)

        min_total_outcome_occurrences = config.getint(
            name,
            "min_total_outcome_occurrences",
            fallback=1 if "binary" in outcome_types else 0,
        )
        min_post_range = config.getfloat(name, "min_post_range", fallback=None)
        keep_most_recent = config.getint(name, "keep_most_recent", fallback=None)

        n_trials = config.getint(name, "n_trials", fallback=None)
        if n_trials is not None:
            warnings.warn(
                "'n_trials' is deprecated and will be removed in a future release. Specify 'min_asks' instead.",
                DeprecationWarning,
            )
            min_asks = n_trials

        return cls(
            lb=lb,
            ub=ub,
            stimuli_per_trial=stimuli_per_trial,
            outcome_types=outcome_types,
            dim=dim,
            model=model,
            generator=generator,
            min_asks=min_asks,
            refit_every=refit_every,
            min_total_outcome_occurrences=min_total_outcome_occurrences,
            min_post_range=min_post_range,
            keep_most_recent=keep_most_recent,
            min_total_tells=min_total_tells,
            name=name,
        )


class SequentialStrategy(object):
    """Runs a sequence of strategies defined by its config

    All getter methods defer to the current strat

    Args:
        strat_list (list[Strategy]): TODO make this nicely typed / doc'd
    """

    def __init__(self, strat_list: List[Strategy]):
        self.strat_list = strat_list
        self._strat_idx = 0
        self._suggest_count = 0

    @property
    def _strat(self):
        return self.strat_list[self._strat_idx]

    def __getattr__(self, name: str):
        # return current strategy's attr if it's not a container attr
        if "strat_list" not in vars(self):
            raise AttributeError("Have no strategies in container, what happened?")
        return getattr(self._strat, name)

    def _make_next_strat(self):
        if (self._strat_idx + 1) >= len(self.strat_list):
            warnings.warn(
                "Ran out of generators, staying on final generator!", RuntimeWarning
            )
            return

        # populate new model with final data from last model
        assert (
            self.x is not None and self.y is not None
        ), "Cannot initialize next strategy; no data has been given!"
        self.strat_list[self._strat_idx + 1].add_data(self.x, self.y)

        self._suggest_count = 0
        self._strat_idx = self._strat_idx + 1

    def gen(self, num_points: int = 1, **kwargs):
        if self._strat.finished:
            self._make_next_strat()
        self._suggest_count = self._suggest_count + num_points
        return self._strat.gen(num_points=num_points, **kwargs)

    def finish(self):
        self._strat.finish()

    @property
    def finished(self):
        return self._strat_idx == (len(self.strat_list) - 1) and self._strat.finished

    def add_data(self, x, y):
        self._strat.add_data(x, y)

    @classmethod
    def from_config(cls, config: Config):
        strat_names = config.getlist("common", "strategy_names", element_type=str)

        # ensure strat_names are unique
        assert len(strat_names) == len(
            set(strat_names)
        ), f"Strategy names {strat_names} are not all unique!"

        strats = []
        for name in strat_names:
            strat = Strategy.from_config(config, str(name))
            strats.append(strat)

        return cls(strat_list=strats)


class AEPsychStrategy(ConfigurableMixin):
    is_finished = False

    def __init__(self, ax_client: AxClient, bounds: torch.Tensor):
        self.ax_client = ax_client
        self.ax_client.experiment.num_asks = 0
        self.bounds = bounds

    @classmethod
    def get_config_options(cls, config: Config, name: Optional[str] = None) -> Dict:
        # TODO: Fix the mypy errors
        strat_names: List[str] = config.getlist("common", "strategy_names", element_type=str)  # type: ignore
        steps = []
        for name in strat_names:
            generator = config.getobj(name, "generator", fallback=AxSobolGenerator)  # type: ignore
            opts = generator.get_config_options(config, name)
            step = AEPsychGenerationStep(**opts)
            steps.append(step)

        # Add an extra step at the end that we can `ask` endlessly.
        final_step = copy(step)
        final_step.completion_criteria = []
        final_step._transition_criteria = []
        final_step.num_trials = -1
        steps.append(final_step)

        parameters = get_parameters(config)
        bounds = get_bounds(config)

        parameter_constraints = config.getlist(
            "common", "par_constraints", element_type=str, fallback=None
        )

        objectives = get_objectives(config)

        seed = config.getint("common", "random_seed", fallback=None)

        strat = GenerationStrategy(steps=steps)
        ax_client = AxClient(strat, random_seed=seed)
        ax_client.create_experiment(
            name="experiment",
            parameters=parameters,
            parameter_constraints=parameter_constraints,
            objectives=objectives,
        )

        return {"ax_client": ax_client, "bounds": bounds}

    @property
    def finished(self) -> bool:
        if self.is_finished or self.strat.optimization_complete:
            return True

        self.strat._maybe_move_to_next_step()

        return len(self.strat._steps) == (self.strat.current_step.index + 1)

    def finish(self):
        self.is_finished = True

    def gen(self, num_points: int = 1):
        x, _ = self.ax_client.get_next_trials(max_trials=num_points)
        self.strat.experiment.num_asks += num_points

        return x

    def complete_new_trial(self, config, outcome):
        _, trial_index = self.ax_client.attach_trial(config)
        self.complete_existing_trial(trial_index, outcome)

    def complete_existing_trial(self, trial_index, outcome):
        self.ax_client.complete_trial(trial_index, outcome)

    @property
    def experiment(self):
        return self.ax_client.experiment

    @property
    def strat(self):
        return self.ax_client.generation_strategy

    @property
    def can_fit(self):
        return (
            self.strat.model is not None
            and len(self.experiment.trial_indices_by_status[TrialStatus.COMPLETED]) > 0
        )

    @property
    def model(self):
        ax_model = self.ax_client.generation_strategy.model
        if not hasattr(ax_model, "surrogate"):
            return None
        aepsych_model = ax_model.model.surrogate.model
        return aepsych_model

    def _warn_on_outcome_mismatch(self):
        if (
            hasattr(self.model, "outcome_type")
            and self.model.outcome_type != "continuous"
        ):
            warnings.warn(
                "Cannot directly plot non-continuous outcomes. Plotting the latent function instead."
            )

    def plot_contours(
        self, density: int = 50, slice_values: Optional[Dict[str, Any]] = None
    ):
        """Plot predictions for a 2-d slice of the parameter space.

        Args:
            density: Number of points along each parameter to evaluate predictions.
            slice_values: A dictionary {name: val} for the fixed values of the
                other parameters. If not provided, then the mean of numeric
                parameters or the mode of choice parameters will be used.
        """
        assert (
            len(self.experiment.parameters) > 1
        ), "plot_contours requires at least 2 parameters! Use 'plot_slice' instead."
        ax_model = self.ax_client.generation_strategy.model
        self._warn_on_outcome_mismatch()

        render(
            interact_contour(
                model=ax_model,
                metric_name="objective",
                density=density,
                slice_values=slice_values,
            )
        )

    def plot_slice(
        self,
        param_name: str,
        density: int = 50,
        slice_values: Optional[Dict[str, Any]] = None,
    ):
        """Plot predictions for a 1-d slice of the parameter space.

        Args:
            param_name: Name of parameter that will be sliced
            density: Number of points along slice to evaluate predictions.
            slice_values: A dictionary {name: val} for the fixed values of the
                other parameters. If not provided, then the mean of numeric
                parameters or the mode of choice parameters will be used.
        """
        self._warn_on_outcome_mismatch()
        ax_model = self.ax_client.generation_strategy.model
        render(
            plot_slice(
                model=ax_model,
                param_name=param_name,
                metric_name="objective",
                density=density,
                slice_values=slice_values,
            )
        )

    def get_pareto_optimal_parameters(self):
        return self.ax_client.get_pareto_optimal_parameters()

    def predict(self, *args, **kwargs):
        """Query the model for posterior mean and variance.; see AEPsychModel.predict."""
        return self.model.predict(self._bounds, *args, **kwargs)

    def predict_probability(self, *args, **kwargs):
        """Query the model in prodbability space for posterior mean and variance.; see AEPsychModel.predict_probability."""
        return self.model.predict(self._bounds, *args, **kwargs)

    def get_max(self, *args, **kwargs):
        """Return the maximum of the modeled function; see AEPsychModel.get_max."""
        return self.model.get_max(self._bounds, *args, **kwargs)

    def get_min(self, *args, **kwargs):
        """Return the minimum of the modeled function; see AEPsychModel.get_min."""
        return self.model.get_min(self._bounds, *args, **kwargs)

    def inv_query(self, *args, **kwargs):
        """Return nearest x such that f(x) = queried y, and also return the
        value of f at that point.; see AEPsychModel.inv_query."""
        return self.model.inv_query(self._bounds, *args, **kwargs)
