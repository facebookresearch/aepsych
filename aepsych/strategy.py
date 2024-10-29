#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import time
import warnings

from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch

from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.generators.sobol_generator import SobolGenerator
from aepsych.models.base import AEPsychMixin, ModelProtocol
from aepsych.utils import _process_bounds, make_scaled_sobol
from aepsych.utils_logging import getLogger
from botorch.exceptions.errors import ModelFittingError

logger = getLogger()


def ensure_model_is_fresh(f:Callable) -> Callable:
    """Decorator to ensure that the model is up-to-date before running a method.

    Args:
        f (Callable): The function to wrap.

    Returns:
        Callable: The wrapped function.
    """
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
        model: Optional[AEPsychMixin] = None,
        refit_every: int = 1,
        min_total_outcome_occurrences: int = 1,
        max_asks: Optional[int] = None,
        keep_most_recent: Optional[int] = None,
        min_post_range: Optional[float] = None,
        name: str = "",
        run_indefinitely: bool = False,
    ) -> None:
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

        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.n: int = 0
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

    def normalize_inputs(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """converts inputs into normalized format for this strategy

        Args:
            x (torch.Tensor): training inputs
            y (torch.Tensor): training outputs

        Returns:
            x (torch.Tensor): training inputs, normalized
            y (torch.Tensor): training outputs, normalized
            n (int): number of observations
        """
        assert (
            x.shape == self.event_shape or x.shape[1:] == self.event_shape
        ), f"x shape should be {self.event_shape} or batch x {self.event_shape}, instead got {x.shape}"

        # Handle scalar y values
        if y.ndim == 0:
            y = y.unsqueeze(0)

        if x.shape == self.event_shape:
            x = x[None, :]

        if self.x is not None:
            x = torch.cat((self.x, x), dim=0)

        if self.y is not None:
            y = torch.cat((self.y, y), dim=0)

        # Ensure the correct dtype
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        n = y.shape[0]

        return x, y, n

    # TODO: allow user to pass in generator options
    @ensure_model_is_fresh
    def gen(self, num_points: int = 1) -> torch.Tensor:
        """Query next point(s) to run by optimizing the acquisition function.

        Args:
            num_points (int, optional): Number of points to query. Defaults to 1.
            Other arguments are forwared to underlying model.

        Returns:
            torch.Tensor: Next set of point(s) to evaluate, [num_points x dim].
        """
        self._count = self._count + num_points
        return self.generator.gen(num_points, self.model)

    @ensure_model_is_fresh
    def get_max(self, constraints: Optional[Mapping[int, List[float]]]  = None, probability_space: bool = False, max_time: Optional[float] = None) -> Tuple[float, torch.Tensor]:
        """Get the maximum value of the acquisition function.
        
        Args:
            constraints (Optional[Mapping[int, List[float]]], optional): Constraints on the input space. Defaults to None.
            probability_space (bool, optional): Whether to return the max in probability space. Defaults to False.
            max_time (Optional[float], optional): Maximum time to run the optimization. Defaults to None.
            
        Returns:
            Tuple[float, torch.Tensor]: The maximum value of the acquisition function and the corresponding input."""
        constraints = constraints or {}
        assert self.model is not None, "model is None! Cannot get the max without a model!"
        return self.model.get_max(
            constraints, probability_space=probability_space, max_time=max_time
        )

    @ensure_model_is_fresh
    def get_min(self, constraints: Optional[Mapping[int, List[float]]]  = None, probability_space: bool = False, max_time: Optional[float] = None) -> Tuple[float, torch.Tensor]:
        """Get the minimum value of the acquisition function.
        
        Args:
            constraints (Optional[Mapping[int, List[float]]], optional): Constraints on the input space. Defaults to None.
            probability_space (bool, optional): Whether to return the min in probability space. Defaults to False.
            max_time (Optional[float], optional): Maximum time to run the optimization. Defaults to None.
            
        Returns:
            Tuple[float, torch.Tensor]: The minimum value of the acquisition function and the corresponding input."""
        constraints = constraints or {}
        assert self.model is not None, "model is None! Cannot get the min without a model!"
        return self.model.get_min(
            constraints, probability_space=probability_space, max_time=max_time
        )

    @ensure_model_is_fresh
    def inv_query(self, y: int, constraints: Optional[Mapping[int, List[float]]]  = None, probability_space: bool = False, max_time: Optional[float] = None) -> Tuple[float, torch.Tensor]:
        """Get the input that corresponds to a given output value.

        Args:
            y (int): The output value to query.
            constraints (Optional[Mapping[int, List[float]]], optional): Constraints on the input space. Defaults to None.
            probability_space (bool, optional): Whether to return the input in probability space. Defaults to False.
            max_time (Optional[float], optional): Maximum time to run the optimization. Defaults to None.

        Returns:
            Tuple[float, torch.Tensor]: The input that corresponds to the given output value and the corresponding output."""
        constraints = constraints or {}
        assert self.model is not None, "model is None! Cannot get the inv_query without a model!"
        return self.model.inv_query(
            y, constraints, probability_space, max_time=max_time
        )

    @ensure_model_is_fresh
    def predict(self, x: torch.Tensor, probability_space: bool = False) -> torch.Tensor:
        """Predict the output of the model at a given input.

        Args:
            x (torch.Tensor): The input to predict the output at.
            Probability_space (bool, optional): Whether to return the output in probability space. Defaults to False.

        Returns:
            torch.Tensor: The predicted output at the given input.
        """
        assert self.model is not None, "model is None! Cannot predict without a model!"
        return self.model.predict(x=x, probability_space=probability_space)

    @ensure_model_is_fresh
    def get_jnd(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get the just-noticeable difference of the model.

        Args:
            *args: Arguments to pass to the model's get_jnd method.
            **kwargs: Keyword arguments to pass to the model's get_jnd method.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: The just-noticeable difference of the model.
        """
        assert self.model is not None, "model is None! Cannot get the get jnd without a model!"
        return self.model.get_jnd(*args, **kwargs)

    @ensure_model_is_fresh
    def sample(self, x: torch.Tensor, num_samples: Optional[int] = None) -> torch.Tensor:
        """Sample the model at a given input.

        Args:
            x (torch.Tensor): The input to sample the model at.
            num_samples (Optional[int], optional): The number of samples to take. Defaults to None.

        Returns:
            torch.Tensor: The samples taken at the given input.
        """
        assert self.model is not None, "model is None! Cannot sample without a model!"
        return self.model.sample(x, num_samples=num_samples)

    def finish(self) -> None:
        """Finish the strategy."""
        self.is_finished = True

    @property
    def finished(self) -> bool:
        """Check if the strategy is finished.

        Returns:
            bool: True if the strategy is finished, False otherwise.
        """
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
            sufficient_outcomes = bool(
                (n_yes_trials >= self.min_total_outcome_occurrences).item()
                and (n_no_trials >= self.min_total_outcome_occurrences).item()
            )
        else:
            sufficient_outcomes = True

        if self.min_post_range is not None:
            assert self.model is not None, "model is None! Cannot predict without a model!"
            fmean, _ = self.model.predict(self.eval_grid, probability_space=True)
            meets_post_range = ((fmean.max() - fmean.min()) >= self.min_post_range).item()
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
    def can_fit(self) -> bool:
        """Check if the strategy can be fitted.

        Returns:
            bool: True if the strategy can be fitted, False otherwise.
        """
        return self.has_model and self.x is not None and self.y is not None

    @property
    def n_trials(self) -> int:
        warnings.warn(
            "'n_trials' is deprecated and will be removed in a future release. Specify 'min_asks' instead.",
            DeprecationWarning,
        )
        return self.min_asks

    def add_data(
        self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """
        Adds new data points to the strategy, and normalizes the inputs.

        Args:
            x (torch.Tensor, np.ndarray): The input data points. Can be a PyTorch tensor or NumPy array.
            y (torch.Tensor, np.ndarray): The output data points. Can be a PyTorch tensor or NumPy array.

        """
        # Necessary as sometimes the data is passed in as numpy arrays or torch tensors.
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64)

        self.x, self.y, self.n = self.normalize_inputs(x, y)
        self._model_is_fresh = False

    def fit(self) -> None:
        """Fit the model to the data."""
        if self.can_fit:
            if self.keep_most_recent is not None:
                try:
                    
                    self.model.fit( # type: ignore
                        self.x[-self.keep_most_recent :], # type: ignore
                        self.y[-self.keep_most_recent :], # type: ignore
                    )
                except ModelFittingError:
                    logger.warning(
                        "Failed to fit model! Predictions may not be accurate!"
                    )
            else:
                try:
                    self.model.fit(self.x, self.y) # type: ignore
                except ModelFittingError:
                    logger.warning(
                        "Failed to fit model! Predictions may not be accurate!"
                    )
        else:
            warnings.warn("Cannot fit: no model has been initialized!", RuntimeWarning)

    def update(self) -> None:
        """Update the model with the most recent data."""
        
        if self.can_fit:
            if self.keep_most_recent is not None:
                try: 
                    self.model.update( # type: ignore
                        self.x[-self.keep_most_recent :], # type: ignore
                        self.y[-self.keep_most_recent :], # type: ignore
                    )
                except ModelFittingError:
                    logger.warning(
                        "Failed to fit model! Predictions may not be accurate!"
                    )
            else:
                try:
                    self.model.update(self.x, self.y) # type: ignore
                except ModelFittingError:
                    logger.warning(
                        "Failed to fit model! Predictions may not be accurate!"
                    )
        else:
            warnings.warn("Cannot fit: no model has been initialized!", RuntimeWarning)

    @classmethod
    def from_config(cls, config: Config, name: str) -> Strategy:
        """Create a strategy from a configuration object.

        Args:
            config (Config): The configuration object.
            name (str): The name of the strategy.

        Returns:
            Strategy: The strategy object.
        """
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

    def __init__(self, strat_list: List[Strategy]) -> None:
        """Initialize the sequential strategy object.
        
        Args:
            strat_list (List[Strategy]): The list of strategies to run.
        """
        self.strat_list = strat_list
        self._strat_idx = 0
        self._suggest_count = 0
        self.x: Optional[torch.Tensor]
        self.y: Optional[torch.Tensor]

    @property
    def _strat(self) -> Strategy:
        """Get the current strategy."""
        return self.strat_list[self._strat_idx]

    def __getattr__(self, name: str) -> Any:
        """Get an attribute of the current strategy if it is not a container attribute."""
        # return current strategy's attr if it's not a container attr
        if "strat_list" not in vars(self):
            raise AttributeError("Have no strategies in container, what happened?")
        return getattr(self._strat, name)

    def _make_next_strat(self) -> None:
        """Switch to the next strategy in the sequence."""
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

    def gen(self, num_points: int = 1, **kwargs) -> torch.Tensor:
        """Generate the next set of points to evaluate.
        
        Args:
            num_points (int, optional): The number of points to generate. Defaults to 1.
            
        Returns:
            torch.Tensor: The next set of points to evaluate."""
        if self._strat.finished:
            self._make_next_strat()
        self._suggest_count = self._suggest_count + num_points
        return self._strat.gen(num_points=num_points, **kwargs)

    def finish(self) -> None:
        """Finish the strategy."""
        self._strat.finish()

    @property
    def finished(self) -> bool:
        """Check if the strategy is finished.

        Returns:
            bool: True if the strategy is finished, False otherwise.
        """
        return self._strat_idx == (len(self.strat_list) - 1) and self._strat.finished

    def add_data(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> None:
        """Add data to the strategy.
        
        Args:
            x (Union[np.ndarray, torch.Tensor]): The input data points.
            y (Union[np.ndarray, torch.Tensor]): The output data points.
        """
        self._strat.add_data(x, y)

    @classmethod
    def from_config(cls, config: Config) -> SequentialStrategy:
        """Create a sequential strategy from a configuration object.
        
        Args:
            config (Config): The configuration object.
            
        Returns:
            SequentialStrategy: The sequential strategy object.
        """
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
