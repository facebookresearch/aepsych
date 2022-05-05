#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import warnings
from typing import Optional, Union

import numpy as np
import torch

from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.generators.sobol_generator import SobolGenerator
from aepsych.models.base import ModelProtocol
from aepsych.utils import _process_bounds, make_scaled_sobol
from aepsych.utils_logging import getLogger

logger = getLogger()


def ensure_model_is_fresh(f):
    def wrapper(self, *args, **kwargs):
        if self.has_model and not self._model_is_fresh:
            if self.x is not None and self.y is not None:
                starttime = time.time()
                if self._count % self.refit_every == 0 or self.refit_every == 1:
                    logger.info("Starting fitting (no warm start)...")
                    # don't warm start
                    self.model.fit(self.x, self.y)
                else:
                    logger.info("Starting fitting (warm start)...")
                    # warm start
                    self.model.update(self.x, self.y)
                logger.info(f"Fitting done, took {time.time()-starttime}")
        self._model_is_fresh = True
        return f(self, *args, **kwargs)

    return wrapper


class Strategy(object):
    def __init__(
        self,
        n_trials: int,
        generator: AEPsychGenerator,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        model: Optional[ModelProtocol] = None,
        dim: Optional[int] = None,
        refit_every: int = 1,
        outcome_type: str = "single_probit",
        min_yes_trials: int = 1,
        min_no_trials: int = 1,
        min_post_range: Optional[float] = None,
        n_eval_points: int = 1000,
        max_trials: Optional[int] = None,
    ):
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)

        self.min_yes_trials = min_yes_trials
        self.min_no_trials = min_no_trials
        self.max_trials = max_trials

        self.min_post_range = min_post_range
        if self.min_post_range is not None:
            assert model is not None, "min_post_range must be None if model is None!"
            self.eval_grid = make_scaled_sobol(
                lb=self.lb, ub=self.ub, size=n_eval_points
            )

        self.x = None
        self.y = None
        self.n_trials = n_trials
        self._count = 0

        # I think this is a correct usage of event_shape
        if type(self.dim) is int:
            self.event_shape = (self.dim,)
        else:
            self.event_shape = self.dim
        self.outcome_type = outcome_type
        self.model = model
        self.refit_every = refit_every
        self._model_is_fresh = False
        self.generator = generator
        self.has_model = self.model is not None
        if self.generator._requires_model:
            assert self.model is not None, f"{self.generator} requires a model!"

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
        if self.outcome_type == "single_probit":
            assert (
                x.shape[-1:] == self.event_shape
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
    def get_max(self, constraints=None):
        constraints = constraints or {}
        return self.model.get_max(constraints)

    @ensure_model_is_fresh
    def get_min(self, constraints=None):
        constraints = constraints or {}
        return self.model.get_min(constraints)

    @ensure_model_is_fresh
    def inv_query(self, y, constraints=None, probability_space=False):
        constraints = constraints or {}
        return self.model.inv_query(y, constraints, probability_space)

    @ensure_model_is_fresh
    def predict(self, x, probability_space=False):
        return self.model.predict(x, probability_space)

    @ensure_model_is_fresh
    def get_jnd(self, *args, **kwargs):
        return self.model.get_jnd(*args, **kwargs)

    @ensure_model_is_fresh
    def sample(self, x, num_samples=None):
        return self.model.sample(x, num_samples=num_samples)

    @property
    def finished(self):
        if self.y is None:  # always need some data before switching strats
            return False

        if self.max_trials is not None and self._count >= self.max_trials:
            return True

        n_yes_trials = (self.y == 1).sum()
        n_no_trials = (self.y == 0).sum()
        if self.min_post_range is not None:
            fmean, _ = self.model.predict(self.eval_grid, probability_space=True)
            meets_post_range = (fmean.max() - fmean.min()) >= self.min_post_range
        else:
            meets_post_range = True
        finished = (
            self._count >= self.n_trials
            and n_yes_trials >= self.min_yes_trials
            and n_no_trials >= self.min_no_trials
            and meets_post_range
        )
        return finished

    @property
    def can_fit(self):
        return self.has_model and self.x is not None and self.y is not None

    def add_data(self, x, y):
        self.x, self.y, self.n = self.normalize_inputs(x, y)
        self._model_is_fresh = False

    def fit(self):
        if self.can_fit:
            self.model.fit(self.x, self.y)
        else:
            warnings.warn("Cannot fit: no model has been initialized!", RuntimeWarning)

    @classmethod
    def from_config(cls, config: Config, name: str):
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

        n_trials = config.getint(name, "n_trials")
        refit_every = config.getint(name, "refit_every", fallback=1)

        lb = config.gettensor(name, "lb")
        ub = config.gettensor(name, "ub")
        dim = config.getint(name, "dim", fallback=None)

        outcome_type = config.get(name, "outcome_type", fallback="single_probit")

        if model is not None and not generator._requires_model:
            if refit_every < n_trials:
                warnings.warn(
                    f"Strategy '{name}' has refit_every < n_trials even though its generator does not require a model. Consider making refit_every = n_trials to speed up point generation.",
                    RuntimeWarning,
                )

        min_yes_trials = config.getint(name, "min_yes_trials", fallback=1)
        min_no_trials = config.getint(name, "min_no_trials", fallback=1)
        min_post_range = config.getfloat(name, "min_post_range", fallback=None)

        return cls(
            lb=lb,
            ub=ub,
            dim=dim,
            model=model,
            generator=generator,
            n_trials=n_trials,
            refit_every=refit_every,
            outcome_type=outcome_type,
            min_yes_trials=min_yes_trials,
            min_no_trials=min_no_trials,
            min_post_range=min_post_range,
        )


class SequentialStrategy(object):
    """Runs a sequence of strategies defined by its config

    All getter methods defer to the current strat

    Args:
        strat_list (list[Strategy]): TODO make this nicely typed / doc'd
    """

    def __init__(self, strat_list):
        self.strat_list = strat_list
        self._strat_idx = 0
        self._suggest_count = 0

    @property
    def _strat(self):
        return self.strat_list[self._strat_idx]

    def __getattr__(self, name):
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

    def gen(self, num_points=1, **kwargs):
        if self._strat.finished:
            self._make_next_strat()
        self._suggest_count = self._suggest_count + num_points
        return self._strat.gen(num_points=num_points, **kwargs)

    @property
    def finished(self):
        return self._strat_idx == (len(self.strat_list) - 1) and self._strat.finished

    def add_data(self, x, y):
        self._strat.add_data(x, y)

    @classmethod
    def from_config(cls, config: Config):
        strat_names = config.getlist("common", "strategy_names", element_type=str)
        strats = []
        for name in strat_names:
            strat = Strategy.from_config(config, str(name))
            strats.append(strat)

        return cls(strat_list=strats)
