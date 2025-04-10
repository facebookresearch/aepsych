#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
import torch
from aepsych.config import Config, ConfigurableMixin
from aepsych.generators.base import AEPsychGenerator
from aepsych.models.base import AEPsychModelMixin
from aepsych.models.utils import get_max, get_min, inv_query
from aepsych.strategy.utils import ensure_model_is_fresh
from aepsych.transforms import (
    ParameterTransformedGenerator,
    ParameterTransformedModel,
    ParameterTransforms,
)
from aepsych.utils import _process_bounds, make_scaled_sobol
from aepsych.utils_logging import getLogger
from botorch.exceptions.errors import ModelFittingError
from botorch.models.transforms.input import ChainedInputTransform

logger = getLogger()


class Strategy(ConfigurableMixin):
    """Object that combines models and generators to generate points to sample."""

    _n_eval_points: int = 1000

    def __init__(
        self,
        generator: AEPsychGenerator | ParameterTransformedGenerator,
        lb: np.ndarray | torch.Tensor,
        ub: np.ndarray | torch.Tensor,
        outcome_types: list[str],
        stimuli_per_trial: int = 1,
        dim: int | None = None,
        min_total_tells: int = 0,
        min_asks: int = 0,
        model: AEPsychModelMixin | None = None,
        use_gpu_modeling: bool = False,
        use_gpu_generating: bool = False,
        refit_every: int = 1,
        min_total_outcome_occurrences: int = 1,
        max_asks: int | None = None,
        keep_most_recent: int | None = None,
        min_post_range: float | None = None,
        name: str = "",
        run_indefinitely: bool = False,
        transforms: ChainedInputTransform = ChainedInputTransform(**{}),
    ) -> None:
        """Initialize the strategy object.

        Args:
            generator (AEPsychGenerator): The generator object that determines how points are sampled.
            lb (torch.Tensor): Lower bounds of the parameters.
            ub (torch.Tensor): Upper bounds of the parameters.
            outcome_types (Sequence[Type[str]]): The types of outcomes that the strategy will generate.
            stimuli_per_trial (int): The number of stimuli per trial, defaults to 1.
            dim (int, optional): The number of dimensions in the parameter space. If None, it is inferred from the size
                of lb and ub.
            min_total_tells (int): The minimum number of total observations needed to complete this strategy.
            min_asks (int): The minimum number of points that should be generated from this strategy.
            model (AEPsychModelMixin, optional): The AEPsych model of the data.
            use_gpu_modeling (bool): Whether to move the model to GPU fitting/predictions, defaults to False.
            use_gpu_generating (bool): Whether to use the GPU for generating points, defaults to False.
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
            transforms (ReversibleInputTransform, optional): Transforms
                to apply parameters. This is immediately applied to lb/ub, thus lb/ub
                should be defined in raw parameter space for initialization. However,
                if the lb/ub attribute are access from an initialized Strategy object,
                it will be returned in transformed space.
        """
        self.is_finished = False

        if run_indefinitely:
            logger.warning(
                f"Strategy {name} will run indefinitely until finish() is explicitly called. Other stopping criteria will be ignored."
            )

        elif min_total_tells > 0 and min_asks > 0:
            logger.warning(
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
                assert (
                    set(outcome_types) == set(model.outcome_type)
                ), f"Strategy outcome types is {outcome_types} but model outcome type is {model.outcome_type}!"

            if use_gpu_modeling:
                if not torch.cuda.is_available():
                    logger.warning(
                        f"GPU requested for model {type(model).__name__}, but no GPU found! Using CPU instead.",
                    )
                    self.model_device = torch.device("cpu")
                else:
                    self.model_device = torch.device("cuda")
                    logger.info(f"Using GPU for modeling with {type(model).__name__}")
            else:
                self.model_device = torch.device("cpu")

        if use_gpu_generating:
            if model is None:
                logger.warning(
                    f"GPU requested for generator {type(generator).__name__} but this generator has no model to move to GPU. Using CPU instead.",
                )
                self.generator_device = torch.device("cpu")
            else:
                if not torch.cuda.is_available():
                    logger.warning(
                        f"GPU requested for generator {type(generator).__name__}, but no GPU found! Using CPU instead.",
                    )
                    self.generator_device = torch.device("cpu")
                else:
                    self.generator_device = torch.device("cuda")
                    logger.info(
                        f"Using GPU for generating with {type(generator).__name__}"
                    )
        else:
            self.generator_device = torch.device("cpu")

        self.run_indefinitely = run_indefinitely
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        self.min_total_outcome_occurrences = min_total_outcome_occurrences
        self.max_asks = max_asks or generator.max_asks
        self.keep_most_recent = keep_most_recent

        self.transforms = transforms

        self.min_post_range = min_post_range
        if self.min_post_range is not None:
            assert model is not None, "min_post_range must be None if model is None!"
            self.eval_grid = make_scaled_sobol(
                lb=self.lb, ub=self.ub, size=self._n_eval_points
            )

        # similar to ub/lb/grid, x is in raw parameter space
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None
        self.n: int = 0
        self.min_asks = min_asks
        self._count = 0
        self.min_total_tells = min_total_tells
        self.stimuli_per_trial = stimuli_per_trial
        self.outcome_types = outcome_types

        if self.stimuli_per_trial == 1:
            self.event_shape: tuple[int, ...] = (self.dim,)

        if self.stimuli_per_trial > 1:
            self.event_shape = (self.dim, self.stimuli_per_trial)

        self.model = model
        self.refit_every = refit_every
        self._model_is_fresh = False
        self.generator = generator
        self.has_model = self.model is not None
        if self.generator._requires_model:
            assert self.model is not None, f"{self.generator} requires a model!"

        if self.min_asks == self.min_total_tells == 0 and self.max_asks is None:
            logger.warning(
                "strategy.min_asks == strategy.min_total_tells == 0. This strategy will not generate any points!",
            )

        self.name = name
        self.bounds = torch.stack([self.lb, self.ub])

    def normalize_inputs(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
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

    @ensure_model_is_fresh
    def gen(self, num_points: int = 1, **kwargs) -> torch.Tensor:
        """Query next point(s) to run by optimizing the acquisition function.

        Args:
            num_points (int): Number of points to query. Defaults to 1.
            Other arguments are forwared to underlying model.
            **kwargs: Kwargs to send to pass to the underlying generator.

        Returns:
            torch.Tensor: Next set of point(s) to evaluate, [num_points x dim].
        """
        original_device = None
        if self.model is not None and self.generator_device.type == "cuda":
            original_device = self.model.device
            self.model.to(self.generator_device)  # type: ignore

        self._count = self._count + num_points
        points = self.generator.gen(num_points, self.model, **kwargs)

        if original_device is not None:
            self.model.to(original_device)  # type: ignore

        return points

    @ensure_model_is_fresh
    def get_max(
        self,
        constraints: Mapping[int, float] | None = None,
        probability_space: bool = False,
        max_time: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the maximum of the modeled function, subject to constraints

        Args:
            constraints (Mapping[int, float], optional): Which parameters to fix at specfic points. Defaults to None.
            probability_space (bool): Whether to return the max in probability space. Defaults to False.
            max_time (float, optional): Maximum time to run the optimization. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the max and its location (argmax).
        """
        assert (
            self.model is not None
        ), "model is None! Cannot get the max without a model!"
        self.model.to(self.model_device)

        val, arg = get_max(
            self.model,
            self.bounds,
            locked_dims=constraints,
            probability_space=probability_space,
            max_time=max_time,
        )

        return val, arg

    @ensure_model_is_fresh
    def get_min(
        self,
        constraints: Mapping[int, float] | None = None,
        probability_space: bool = False,
        max_time: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the minimum of the modeled function, subject to constraints

        Args:
            constraints (Mapping[int, float], optional): Which parameters to fix at specific points. Defaults to None.
            probability_space (bool): Whether to return the min in probability space. Defaults to False.
            max_time (float, optional): Maximum time to run the optimization. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the min and its location (argmin).
        """
        assert (
            self.model is not None
        ), "model is None! Cannot get the min without a model!"
        self.model.to(self.model_device)

        val, arg = get_min(
            self.model,
            self.bounds,
            locked_dims=constraints,
            probability_space=probability_space,
            max_time=max_time,
        )

        return val, arg

    @ensure_model_is_fresh
    def inv_query(
        self,
        y: int,
        constraints: Mapping[int, float] | None = None,
        probability_space: bool = False,
        max_time: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the input that corresponds to a given output value.

        Args:
            y (int): The output value.
            constraints (Mapping[int, list[float]], optional): Which parameters to fix at specific points. Defaults to None.
            probability_space (bool): Whether to return the input in probability space. Defaults to False.
            max_time (float, optional): Maximum time to run the optimization. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The input that corresponds to the given output value and the corresponding output.
        """
        assert (
            self.model is not None
        ), "model is None! Cannot get the inv_query without a model!"
        self.model.to(self.model_device)

        val, arg = inv_query(
            model=self.model,
            y=y,
            bounds=self.bounds,
            locked_dims=constraints,
            probability_space=probability_space,
            max_time=max_time,
        )

        return val, arg

    @ensure_model_is_fresh
    def predict(
        self, x: torch.Tensor, probability_space: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the output value(s) for the given input(s).

        Args:
            x (torch.Tensor): The input value(s).
            probability_space (bool): Whether to return the output in probability space. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at query points.
        """
        assert self.model is not None, "model is None! Cannot predict without a model!"
        self.model.to(self.model_device)
        return self.model.predict(x=x, probability_space=probability_space)

    @ensure_model_is_fresh
    def sample(self, x: torch.Tensor, num_samples: int = 1000) -> torch.Tensor:
        """Sample the output value(s) for the given input(s).

        Args:
            x (torch.Tensor): The input value(s).
            num_samples (int): The number of samples to generate. Defaults to 1000.

        Returns:
            torch.Tensor: The sampled output value(s).
        """
        assert self.model is not None, "model is None! Cannot sample without a model!"
        self.model.to(self.model_device)
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

        if self.y is None:  # always need some data before switching strats
            return False

        if self.max_asks is not None and self._count >= self.max_asks:
            return True

        if hasattr(self.generator, "finished"):  # defer to generator if possible
            return self.generator.finished

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
            assert (
                self.model is not None
            ), "model is None! Cannot predict without a model!"
            fmean, _ = self.model.predict(self.eval_grid, probability_space=True)
            meets_post_range = bool(
                ((fmean.max() - fmean.min()) >= self.min_post_range).item()
            )
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

    def pre_warm_model(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Adds new data points to the strategy, and normalizes the inputs.
        We speceifically disregard the n return value of normalize_inputs here in order
        to stop warm start data from affecting the trials run length.

        Args:
            x torch.Tensor: The input data points.
            y torch.Tensor: The output data points.

        """
        # warming the model shouldn't affect strategy.n
        self.x, self.y, n = self.normalize_inputs(x, y)
        self._model_is_fresh = False

    def add_data(
        self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> None:
        """
        Adds new data points to the strategy, and normalizes the inputs.

        Args:
            x (np.ndarray | torch.Tensor): The input data points. Can be a PyTorch tensor or NumPy array.
            y (np.ndarray | torch.Tensor): The output data points. Can be a PyTorch tensor or NumPy array.

        """
        # Necessary as sometimes the data is passed in as numpy arrays or torch tensors.
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64)

        self.x, self.y, self.n = self.normalize_inputs(x, y)
        self._model_is_fresh = False

    def fit(self) -> None:
        """Fit the model."""
        if self.can_fit:
            self.model.to(self.model_device)  # type: ignore
            if self.keep_most_recent is not None:
                try:
                    self.model.fit(  # type: ignore
                        self.x[-self.keep_most_recent :],  # type: ignore
                        self.y[-self.keep_most_recent :],  # type: ignore
                    )
                except ModelFittingError:
                    logger.warning(
                        "Failed to fit model! Predictions may not be accurate!"
                    )
            else:
                try:
                    self.model.fit(self.x, self.y)  # type: ignore
                except ModelFittingError:
                    logger.warning(
                        "Failed to fit model! Predictions may not be accurate!"
                    )
        else:
            warnings.warn("Cannot fit: no model has been initialized!", RuntimeWarning)

    def update(self) -> None:
        """Update the model."""
        if self.can_fit:
            self.model.to(self.model_device)  # type: ignore
            if self.keep_most_recent is not None:
                try:
                    self.model.update(  # type: ignore
                        self.x[-self.keep_most_recent :],  # type: ignore
                        self.y[-self.keep_most_recent :],  # type: ignore
                    )
                except ModelFittingError:
                    logger.warning(
                        "Failed to fit model! Predictions may not be accurate!"
                    )
            else:
                try:
                    self.model.update(self.x, self.y)  # type: ignore
                except ModelFittingError:
                    logger.warning(
                        "Failed to fit model! Predictions may not be accurate!"
                    )
        else:
            warnings.warn("Cannot fit: no model has been initialized!", RuntimeWarning)

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Return a dictionary of the relevant options to initialize this class from the
        config, even if it is outside of the named section. By default, this will look
        for options in name based on the __init__'s arguments/defaults.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): Primary section to look for options for this class and
                the name to infer options from other sections in the config.
            options (dict[str, Any], optional): Options to override from the config,
                defaults to None.


        Return:
            dict[str, Any]: A dictionary of options to initialize this class.
        """
        if name is None:
            raise ValueError(
                "Strategy cannot be initialized from a config without providing a name!"
            )

        options = super().get_config_options(config=config, name=name, options=options)

        # Override transforms
        options["transforms"] = ParameterTransforms.from_config(config)

        # Rebuild generator
        gen_name = config.get(name, "generator")
        options["generator"] = ParameterTransformedGenerator.from_config(
            config, gen_name, options={"transforms": options["transforms"]}
        )
        options["use_gpu_generating"] = config.getboolean(
            options["generator"]._base_obj.__class__.__name__, "use_gpu", fallback=False
        )

        # Rebuild model
        model_cls = config.getobj(name, "model", fallback=None)
        if model_cls is not None:
            model = ParameterTransformedModel.from_config(
                config,
                model_cls.__name__,
                options={"transforms": options["transforms"]},
            )
            use_gpu_modeling = config.getboolean(
                model._base_obj.__class__.__name__, "use_gpu", fallback=False
            )

            if use_gpu_modeling:
                model.cuda()

            options["model"] = model
            options["use_gpu_modeling"] = use_gpu_modeling
        else:
            options["model"] = None
            options["use_gpu_modeling"] = False

        if options["model"] is not None and not options["generator"]._requires_model:
            if options["refit_every"] < options["min_asks"]:
                logger.warning(
                    f"Strategy '{name}' has refit_every < min_asks even though its generator does not require a model. Consider making refit_every = min_asks to speed up point generation.",
                )

        options["min_total_outcome_occurrences"] = config.getint(
            name,
            "min_total_outcome_occurrences",
            fallback=1 if "binary" in options["outcome_types"] else 0,
        )

        options["name"] = name

        return options
