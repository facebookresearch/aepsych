#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional, Union
from aepsych.generators.optimize_acqf_generator import OptimizeAcqfGenerator

import numpy as np
import torch

from aepsych.generators.base import AEPsychGenerator
from aepsych.models.base import AEPsychModel
from aepsych.models.gp_classification import GPClassificationModel
from aepsych.utils import _process_bounds, make_scaled_sobol


def ensure_model_is_fresh(f):
    def wrapper(self, *args, **kwargs):
        if not self._model_is_fresh:
            if self._count % self.refit_every == 0 or self.refit_every == 1:
                # don't warm start
                self.model.fit(self.x, self.y)
            else:
                # warm start
                self.model.update(self.x, self.y)
        self._model_is_fresh = True
        return f(self, *args, **kwargs)

    return wrapper


class Strategy(object):
    has_model = False

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        n_trials: int = -1,
        dim: Optional[int] = None,
        outcome_type: str = "single_probit",
    ):
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)

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

    @property
    def finished(self):
        return self._count >= self.n_trials

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

    def add_data(self, x, y):
        self.x, self.y, self.n = self.normalize_inputs(x, y)


class ModelWrapperStrategy(Strategy):
    """Wraps a Model into a strategy by forwarding calls and
    storing data

    Args:
        model (Model): the underlying model

    """

    has_model = True

    def __init__(
        self,
        model: AEPsychModel,
        generator: AEPsychGenerator,
        n_trials: int,
        stopping_threshold: Optional[float] = None,
        refit_every: int = 1,
    ):
        super().__init__(
            lb=model.lb,
            ub=model.ub,
            n_trials=n_trials,
            dim=model.dim,
            outcome_type=model.outcome_type,
        )
        self.model = model
        self.last_best = None
        self.stopping_threshold = stopping_threshold
        self.refit_every = refit_every
        self._model_is_fresh = False
        self.generator = generator

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
    def get_max(self):
        return self.model.get_max()

    @ensure_model_is_fresh
    def get_min(self):
        return self.model.get_min()

    @ensure_model_is_fresh
    def inv_query(self, y, constraints, probability_space=False):
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
        if self.stopping_threshold is None:
            return super().finished
        else:
            return self.__finished_with_thresh()

    @ensure_model_is_fresh
    def __finished_with_thresh(self):
        # if we actually have a threshold, update the model and criterion
        if self.last_best is None:
            self.last_best = self.model.best()
        else:
            current_best = self.model.best()
            l_2 = np.linalg.norm(self.last_best - current_best)
            self.last_best = current_best
            return l_2 < self.stopping_threshold

    def add_data(self, x, y):
        super().add_data(x, y)
        self._model_is_fresh = False

    @classmethod
    def from_config(cls, config):
        classname = cls.__name__
        model_cls = config.getobj("experiment", "model", fallback=GPClassificationModel)

        model = model_cls.from_config(config)
        n_trials = config.getint(classname, "n_trials")
        stopping_threshold = config.getfloat(
            classname, "stopping_threshold", fallback=None
        )
        refit_every = config.getint(classname, "refit_every", fallback=1)

        gen_cls = config.getobj(
            "experiment", "generator", fallback=OptimizeAcqfGenerator
        )
        generator = gen_cls.from_config(config)

        return cls(
            model=model,
            generator=generator,
            n_trials=n_trials,
            stopping_threshold=stopping_threshold,
            refit_every=refit_every,
        )


class SobolStrategy(Strategy):
    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        n_trials: int,
        dim: int = None,
        outcome_type: str = "single_probit",
        seed: Optional[int] = None,
    ):
        super().__init__(lb=lb, ub=ub, dim=dim, outcome_type=outcome_type)
        if n_trials <= 0:
            warnings.warn(
                "SobolStrategy was initialized with n_trials <= 0; it will not generate any points!"
            )

        if n_trials > 0:
            self.points = make_scaled_sobol(lb=lb, ub=ub, size=n_trials, seed=seed)
        else:
            self.points = np.array([])

        self.n_trials = n_trials
        self._count = 0
        self.seed = seed

    def gen(self, num_points=1, **kwargs):
        if self._count + num_points > self.n_trials:
            warnings.warn(
                f"Requesting more points ({num_points}) than"
                + f"this sobol sequence has remaining ({self.n_trials-self._count})!"
                + "Giving as many as we have."
            )
            candidates = self.points[self._count :]
        else:
            candidates = self.points[self._count : self._count + num_points]
        self._count = self._count + num_points
        return candidates

    @classmethod
    def from_config(cls, config):
        classname = cls.__name__
        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = config.gettensor(classname, "dim", fallback=None)
        lb, ub, dim = _process_bounds(lb, ub, dim)
        n_trials = config.getint(classname, "n_trials")
        outcome_type = config.get(classname, "outcome_type", fallback="single_probit")
        seed = config.getfloat(classname, "seed", fallback=None)
        return cls(lb, ub, n_trials, dim, outcome_type, seed)


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
    def from_config(cls, config):

        init_strat_cls = config.getobj(
            "experiment", "init_strat_cls", fallback=SobolStrategy
        )
        opt_strat_cls = config.getobj(
            "experiment", "opt_strat_cls", fallback=ModelWrapperStrategy
        )

        init_strat = init_strat_cls.from_config(config)
        opt_strat = opt_strat_cls.from_config(config)

        return cls([init_strat, opt_strat])


class EpsilonGreedyModelWrapperStrategy(ModelWrapperStrategy):
    def __init__(self, epsilon=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    @classmethod
    def from_config(cls, config):
        classname = cls.__name__
        obj = super().from_config(config)
        obj.epsilon = config.getfloat(classname, "epsilon", fallback=0)
        return obj

    def gen(self, num_points=1, *args, **kwargs):
        if num_points > 1:
            raise NotImplementedError("Epsilon-greedy batched gen is not implemented!")
        if np.random.uniform() < self.epsilon:
            self._count = self._count + num_points
            return np.random.uniform(low=self.lb, high=self.ub)
        else:
            return super().gen(*args, **kwargs)
