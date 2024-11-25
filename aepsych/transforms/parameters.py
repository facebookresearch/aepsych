#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ast
import warnings
from abc import ABC
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch
from aepsych.config import Config, ConfigurableMixin
from aepsych.generators.base import AEPsychGenerator
from aepsych.models.base import AEPsychMixin, ModelProtocol
from aepsych.transforms.ops import Fixed, Log10Plus, NormalizeScale, Round
from aepsych.transforms.ops.base import Transform
from aepsych.utils import get_bounds
from botorch.acquisition import AcquisitionFunction
from botorch.models.transforms.input import ChainedInputTransform
from botorch.posteriors import Posterior

_TRANSFORMABLE = [
    "lb",
    "ub",
    "points",
    "window",
]


class ParameterTransforms(ChainedInputTransform, ConfigurableMixin):
    """
    Holds set of transformations to be applied to parameters. The ParameterTransform
    objects can be used by themselves to transform values or can be passed to Generator
    or Model wrappers to consistently transform parameters. ParameterTransforms can
    transform values into transformed space and also untransform values from transformed
    space back into raw space.
    """

    def __init__(
        self,
        **transforms: Transform,
    ) -> None:
        fixed_values = []
        fixed_indices = []
        transform_keys = list(transforms.keys())
        for key in transform_keys:
            if isinstance(transforms[key], Fixed):
                transform = transforms.pop(key)
                fixed_values += transform.values.tolist()
                fixed_indices += transform.indices.tolist()

        if len(fixed_values) > 0:
            # Combine Fixed parameters
            transforms["_CombinedFixed"] = Fixed(
                indices=fixed_indices, values=fixed_values
            )

        super().__init__(**transforms)

    def _temporary_reshape(func: Callable) -> Callable:
        # Decorator to reshape tensors to the expected 2D shape, even if the input was
        # 1D or 3D and after the transform reshape it back to the original.
        def wrapper(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
            squeeze = False
            if len(X.shape) == 1:  # For 1D inputs, primarily for transforming arguments
                X = X.unsqueeze(0)
                squeeze = True

            reshape = False
            if len(X.shape) > 2:  # For multi stimuli experiments
                batch, dim, stim = X.shape
                X = X.swapaxes(-2, -1).reshape(-1, dim)
                reshape = True

            X = func(self, X, **kwargs)

            if reshape:
                X = X.reshape(batch, stim, -1).swapaxes(-1, -2)

            if squeeze:  # Not actually squeeze since we still want one dim
                X = X[0]

            return X

        return wrapper

    @_temporary_reshape
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        r"""Transform the inputs to a model.

        Individual transforms are applied in sequence.

        Args:
            X (torch.Tensor): A tensor of inputs. Either `[dim]`, `[batch, dim]`, or
            `[batch, dim, stimuli]`.

        Returns:
            torch.Tensor: A tensor of transformed inputs with the same shape as the
                input.
        """
        return super().transform(X)

    @_temporary_reshape
    def untransform(self, X: torch.Tensor) -> torch.Tensor:
        r"""Un-transform the inputs to a model.

        Un-transforms of the individual transforms are applied in reverse sequence.

        Args:
            X (torch.Tensor): A tensor of inputs. Either `[dim]`, `[batch, dim]`, or
                `[batch, dim, stimuli]`.

        Returns:
            torch.Tensor: A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        return super().untransform(X)

    @_temporary_reshape
    def transform_bounds(
        self, X: torch.Tensor, bound: Optional[Literal["lb", "ub"]] = None
    ) -> torch.Tensor:
        r"""Transform bounds of a parameter.

        Individual transforms are applied in sequence. Then an adjustment is applied to
        ensure the bounds are correct.

        Args:
            X (torch.Tensor): A tensor of inputs. Either `[dim]` or `[2, dim]`.
            bound: (Literal["lb", "ub"], optional): Which bound this is when
                transforming. If not set, assumes both bounds are given at once

        Returns:
            torch.Tensor: Tensor of the same shape as the input transformed.
        """
        for tf in self.values():
            X = tf.transform_bounds(X, bound=bound)

        return X

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a dictionary of transforms in the order that they should be in, this
        dictionary can be used to initialize a ParameterTransforms.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): Name is ignored as transforms for all parameters will
                be made. Maintained for API conformity.
            options (Dict[str, Any], optional): options is ignored as all transforms
                will be reinitialized to ensure it is in the right order. To create
                transforms in an arbitrary order, initialize from the __init__.


        Return:
            Dict[str, Any]: A dictionary of transforms to initialize this class.
        """
        if options is not None:
            warnings.warn(
                "options argument is set but we will ignore it to create an entirely new options dictionary to ensure transforms are applied in the right order."
            )

        transform_options = {}
        transform_options["bounds"] = get_bounds(config)

        parnames = config.getlist("common", "parnames", element_type=str)

        # This is the "options" dictionary, transform options is only for maintaining the right transforms
        transform_dict: Dict[str, ChainedInputTransform] = {}
        for par in parnames:
            # This is the order that transforms are potentially applied, order matters

            try:
                par_type = config[par]["par_type"]
            except KeyError:  # Probably because par doesn't have its own section
                par_type = "continuous"

            # Integer or binary variable
            if par_type in ["integer", "binary"]:
                round = Round.from_config(
                    config=config, name=par, options=transform_options
                )

                # Transform bounds
                transform_options["bounds"] = round.transform_bounds(
                    transform_options["bounds"]
                )
                transform_dict[f"{par}_Round"] = round

            if par_type == "fixed":
                fixed = Fixed.from_config(
                    config=config, name=par, options=transform_options
                )

                # We don't mess with bounds since we don't want to modify indices
                transform_dict[f"{par}_Fixed"] = fixed

            # Log scale
            if config.getboolean(par, "log_scale", fallback=False):
                log10 = Log10Plus.from_config(
                    config=config, name=par, options=transform_options
                )

                # Transform bounds
                transform_options["bounds"] = log10.transform(
                    transform_options["bounds"]
                )
                transform_dict[f"{par}_Log10Plus"] = log10

            # Normalize scale (defaults true)
            if config.getboolean(
                par, "normalize_scale", fallback=True
            ) and par_type not in ["binary", "fixed"]:
                normalize = NormalizeScale.from_config(
                    config=config, name=par, options=transform_options
                )

                # Transform bounds
                transform_options["bounds"] = normalize.transform(
                    transform_options["bounds"]
                )
                transform_dict[f"{par}_NormalizeScale"] = normalize

        return transform_dict


class ParameterTransformWrapper(ABC):
    """
    Abstract base class for parameter transform wrappers. __getattr__ is overridden to
    allow base object attributes to be surfaced smoothly. Methods that require the
    transforms should be overridden in the wrapper class to apply the transform
    operations.
    """

    transforms: ChainedInputTransform
    _base_obj: object = None

    def __getattr__(self, name):
        return getattr(self._base_obj, name)


class ParameterTransformedGenerator(ParameterTransformWrapper, ConfigurableMixin):
    """
    Wraps a generator such that parameter inputs are transformed and generated
    parameters are untransformed back to the raw parameter space.
    """

    _base_obj: AEPsychGenerator

    def __init__(
        self,
        generator: Union[Type, AEPsychGenerator],
        transforms: ChainedInputTransform = ChainedInputTransform(**{}),
        **kwargs: Any,
    ) -> None:
        r"""Wraps a Generator with parameter transforms. This will transform any relevant
        generator arguments (e.g., bounds) to be transformed into the transformed space
        and ensure all generator outputs to be untransformed into raw space. The wrapper
        surfaces critical components of the API of the generator such that the wrapper
        can be used much like the raw generator.

        Bounds are returned in the transformed space, this is necessary to handle
        parameters that would not have sensible raw parameter space. If bounds are
        manually set (e.g., `Wrapper(**kwargs).lb = lb)`, ensure that they are
        correctly transformed and in a correctly shaped Tensor. If the bounds are
        being set in init (e.g., `Wrapper(Type, lb=lb, ub=ub)`, `lb` and `ub`
        should be in the raw parameter space.

        The object's name will be ParameterTransformed<Generator.__name__>.

        Args:
            model (Type | AEPsychGenerator): Generator to wrap, this could either be a
                completely initialized generator or just the generator class. An
                initialized generator is expected to have been initialized in the
                transformed parameter space (i.e., bounds are transformed). If a
                generator class is passed, **kwargs will be used to initialize the
                generator, note that the bounds are expected to be in raw parameter
                space, thus the transforms are applied to it.
            transforms (ChainedInputTransform, optional): A set of transforms to apply
                to parameters of this generator. If no transforms are passed, it will
                default to an identity transform.
            **kwargs: Keyword arguments to pass to the model to initialize it if model
                is a class.
        """
        # Figure out what we need to do with generator
        if isinstance(generator, type):
            if "lb" in kwargs:
                kwargs["lb"] = transforms.transform(kwargs["lb"].to(torch.float64))
            if "ub" in kwargs:
                kwargs["ub"] = transforms.transform(kwargs["ub"].to(torch.float64))
            _base_obj = generator(**kwargs)
        else:
            _base_obj = generator

        self._base_obj = _base_obj
        self.transforms = transforms

        # This lets us emit we're the class we're wrapping
        self.__class__ = type(
            f"ParameterTransformed{_base_obj.__class__.__name__}",
            (self.__class__, _base_obj.__class__),
            {},
        )

        # Copy all of the class attributes with defaults from _base_obj
        # and don't just let it use AEPsychGenerator defaults
        self._requires_model = self._base_obj._requires_model
        self.stimuli_per_trial = self._base_obj.stimuli_per_trial
        self.max_asks = self._base_obj.max_asks

    def gen(
        self, num_points: int = 1, model: Optional[AEPsychMixin] = None
    ) -> torch.Tensor:
        r"""Query next point(s) to run from the generator and return them untransformed.

        Args:
            num_points (int): Number of points to query, defaults to 1.
            model (AEPsychMixin, optional): The model to use to generate points, can be
                None if no model is needed.
        Returns:
            torch.Tensor: Next set of point(s) to evaluate, `[num_points x dim]` or
            `[num_points x dim x stimuli_per_trial]` if `self.stimuli_per_trial != 1`,
            which will be untransformed.
        """
        x = self._base_obj.gen(num_points, model)
        return self.transforms.untransform(x)

    def _get_acqf_options(self, acqf: AcquisitionFunction, config: Config):
        return self._base_obj._get_acqf_options(acqf, config)

    @property
    def acqf(self) -> AcquisitionFunction | None:
        return self._base_obj.acqf

    @acqf.setter
    def acqf(self, value: AcquisitionFunction):
        self._base_obj.acqf = value

    @property
    def acqf_kwargs(self) -> dict | None:
        return self._base_obj.acqf_kwargs

    @acqf_kwargs.setter
    def acqf_kwargs(self, value: dict):
        self._base_obj.acqf_kwargs = value

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a dictionary of the relevant options to initialize a generator wrapped
        with a parameter transform.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): Strategy to look for the Generator and find options for.
            options (Dict[str, Any], optional): Options to override from the config.

        Returns:
            Dict[str, Any]: A diciontary of options to initialize this class with.
        """
        if options is None:
            options = {}
            options["transforms"] = ParameterTransforms.from_config(config)
        else:
            # Check if there's a transform already if so save it to it persists over copying
            if "transforms" in options:
                transforms = options["transforms"]
            else:
                transforms = ParameterTransforms.from_config(config)

            options = deepcopy(options)
            options["transforms"] = transforms

        if name is None:
            raise ValueError("name of strategy must be set to initialize a generator")
        else:
            gen_cls = config.getobj(name, "generator")

        # Transform config
        transformed_config = transform_options(config, options["transforms"])

        options["generator"] = gen_cls.from_config(transformed_config)

        return options


class ParameterTransformedModel(ParameterTransformWrapper, ConfigurableMixin):
    """
    Wraps a model such that it can work entirely in transformed space by transforming
    all parameter inputs (e.g., training data) to the model. This wrapper also
    untransforms any outputs from the model back to raw parameter space.
    """

    _base_obj: ModelProtocol

    def __init__(
        self,
        model: Union[Type, ModelProtocol],
        transforms: ChainedInputTransform = ChainedInputTransform(**{}),
        **kwargs: Any,
    ) -> None:
        """Wraps a Model with parameter transforms. This will transform any relevant
        model arguments (e.g., bounds) and model data (e.g., training data, x) to be
        transformed into the transformed space. The wrapper surfaces the API of the
        raw model such that the wrapper can be used like a raw model.

        Bounds are returned in the transformed space, this is necessary to handle
        parameters that would not have sensible raw parameter space. If bounds are
        manually set (e.g., `Wrapper(**kwargs).lb = lb)`, ensure that they are
        correctly transformed and in a correctly shaped Tensor. If the bounds are
        being set in init (e.g., `Wrapper(Type, lb=lb, ub=ub)`, `lb` and `ub`
        should be in the raw parameter space.

        The object's name will be ParameterTransformed<Model.__name__>.

        Args:
            model (Union[Type, ModelProtocol]): Model to wrap, this could either be a
                completely initialized model or just the model class. An initialized
                model is expected to have been initialized in the transformed
                parameter space (i.e., bounds are transformed). If a model class is
                passed, **kwargs will be used to initialize the model. Note that the
                bounds in this case are expected to be in raw parameter space, thus the
                transforms are applied to it.
            transforms (ChainedInputTransform, optional): A set of transforms to apply
                to parameters of this model. If no transforms are passed, it will
                default to an identity transform.
            **kwargs: Keyword arguments to be passed to the model if the model is a
                class.
        """
        # Alternative instantiation method for analysis (and not live)
        if isinstance(model, type):
            if "lb" in kwargs:
                kwargs["lb"] = transforms.transform(kwargs["lb"].to(torch.float64))
            if "ub" in kwargs:
                kwargs["ub"] = transforms.transform(kwargs["ub"].to(torch.float64))
            _base_obj = model(**kwargs)
        else:
            _base_obj = model

        self._base_obj = _base_obj
        self.transforms = transforms

        # This lets us emit we're the class we're wrapping
        self.__class__ = type(
            f"ParameterTransformed{_base_obj.__class__.__name__}",
            (self.__class__, _base_obj.__class__),
            {},
        )

    @staticmethod
    def _promote_1d(func: Callable) -> Callable:
        # Decorator to reshape model Xs into 2d if they're 1d. Assumes that the Tensor
        # is the first argument
        def wrapper(self, *args, **kwargs) -> torch.Tensor:
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                x = args[0]
                if len(x.shape) == 1:
                    x = x.unsqueeze(-1)
                return func(self, x, *args[1:], **kwargs)

            else:
                key, value = list(kwargs.items())[0]
                if isinstance(value, torch.Tensor) and len(value.shape) == 1:
                    kwargs[key] = value.unsqueeze(-1)

                return func(self, **kwargs)

        return wrapper

    @_promote_1d
    def predict(
        self, x: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Query the model on its posterior given transformed x.

        Args:
            x (torch.Tensor): Points at which to predict from the model, which will
                be transformed.
            **kwargs: Keyword arguments to pass to the model.predict() call.

        Returns:
            Union[Tensor, Tuple[Tensor]]: At least one Tensor will be returned.
        """
        x = self.transforms.transform(x)
        return self._base_obj.predict(x, **kwargs)

    @_promote_1d
    def predict_probability(
        self, x: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Query the model on its posterior given transformed x and return units in
        response probability space.

        Args:
            x (torch.Tensor): Points at which to predict from the model, which will be
                transformed.
            **kwargs: Keyword arguments to pass to the model.predict() call.

        Returns:
            Union[Tensor, Tuple[Tensor]]: At least one Tensor will be returned.
        """
        x = self.transforms.transform(x)
        return self._base_obj.predict_probability(x, **kwargs)

    @_promote_1d
    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample from underlying model given transformed x.

        Args:
            x (torch.Tensor): Points at which to sample, which will be transformed.
            num_samples (int): Number of samples to return.

        Returns:
            torch.Tensor: Posterior samples [num_samples x dim]
        """
        x = self.transforms.transform(x)
        return self._base_obj.sample(x, num_samples)

    def posterior(self, X: torch.Tensor, **kwargs) -> Posterior:
        """Return the model's posterior given a transformed X. Notice that this specific
        method requires transformed inputs.

        Args:
            X (torch.Tensor): The points to evaluate, which will be transformed.
            **kwargs: The keyword arguments to pass to the underlying model's posterior
                method.

        Returns:
            Posterior: The posterior of the model.
        """
        # This ensures X is a tensor with the right shape and dtype (this seemingly
        # does nothing, but it somehow solves test errors).
        X = torch.Tensor(X)
        return self._base_obj.posterior(X=X, **kwargs)

    def dim_grid(self, gridsize: int = 30) -> torch.Tensor:
        """Returns an untransformed grid based on the model's bounds and dimensionality.

        Args:
            gridsize (int): How many points to form the grid with, defaults to
            30.

        Returns:
            Tensor: A grid based on the model's bounds and dimensionality with the
                number of points requested, which will be untransformed.
        """
        grid = self._base_obj.dim_grid(gridsize)
        return self.transforms.untransform(grid)

    @_promote_1d
    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs) -> None:
        """Fit underlying model.

        Args:
            train_x (torch.Tensor): Inputs to fit on.
            train_y (torch.Tensor): Responses to fit on.
            warmstart_hyperparams (bool): Whether to reuse the previous hyperparameters
                (True) or fit from scratch (False). Defaults to False.
            warmstart_induc (bool): Whether to reuse the previous inducing points or fit
                from scratch (False). Defaults to False.
            **kwargs: Keyword arguments to pass to the underlying model's fit method.
        """
        train_x = self.transforms.transform(train_x)
        self._base_obj.fit(train_x, train_y, **kwargs)

    @_promote_1d
    def update(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs) -> None:
        """Perform a warm-start update of the model from previous fit.

        Args:
            train_x (torch.Tensor): Inputs to fit on.
            train_y (torch.Tensor): Responses to fit on.
            **kwargs: Keyword arguments to pass to the underlying model's fit method.
        """
        train_x = self.transforms.transform(train_x)
        self._base_obj.update(train_x, train_y, **kwargs)

    def get_max(
        self,
        locked_dims: Optional[Mapping[int, List[float]]] = None,
        probability_space: bool = False,
        n_samples: int = 1000,
        max_time: Optional[float] = None,
    ) -> Tuple[float, torch.Tensor]:
        """Return the maximum of the modeled function, subject to constraints

        Args:
            locked_dims (Mapping[int, List[float]], optional): Dimensions to fix, so that the
                inverse is along a slice of the full surface. Defaults to None.
            probability_space (bool): Is y (and therefore the returned nearest_y) in
                probability space instead of latent function space? Defaults to False.
            n_samples (int): number of coarse grid points to sample for optimization estimate.
                Defaults to 1000.
            max_time (float, optional): Maximum time to spend optimizing. Defaults to None.

        Returns:
            Tuple[float, torch.Tensor]: Tuple containing the max and its untransformed
                location (argmax).
        """
        max_, loc = self._base_obj.get_max(  # type: ignore
            locked_dims=locked_dims,
            probability_space=probability_space,
            n_samples=n_samples,
            max_time=max_time,
        )
        loc = self.transforms.untransform(loc)

        return max_, loc

    def get_min(
        self,
        locked_dims: Optional[Mapping[int, List[float]]] = None,
        probability_space: bool = False,
        n_samples: int = 1000,
        max_time: Optional[float] = None,
    ) -> Tuple[float, torch.Tensor]:
        """Return the minimum of the modeled function, subject to constraints

        Args:
            locked_dims (Mapping[int, List[float]], optional): Dimensions to fix, so that the
                inverse is along a slice of the full surface.
            probability_space (bool): Is y (and therefore the returned nearest_y) in
                probability space instead of latent function space? Defaults to False.
            n_samples (int): number of coarse grid points to sample for optimization estimate.
                Defaults to 1000.
            max_time (float, optional): Maximum time to spend optimizing. Defaults to None.

        Returns:
            Tuple[float, torch.Tensor]: Tuple containing the min and its untransformed location (argmin).
        """
        min_, loc = self._base_obj.get_min(  # type: ignore
            locked_dims=locked_dims,
            probability_space=probability_space,
            n_samples=n_samples,
            max_time=max_time,
        )
        loc = self.transforms.untransform(loc)

        return min_, loc

    def inv_query(
        self,
        y: float,
        locked_dims: Optional[Mapping[int, List[float]]] = None,
        probability_space: bool = False,
        n_samples: int = 1000,
        max_time: Optional[float] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[float, torch.Tensor]:
        """Query the model inverse.

        Return nearest untransformed x such that f(x) = queried y, and also return the
            value of f at that point.

        Args:
            y (float): Points at which to find the inverse.
            locked_dims (Mapping[int, List[float]], optional): Dimensions to fix, so that the
                inverse is along a slice of the full surface.
            probability_space (bool): Is y (and therefore the returned nearest_y) in
                probability space instead of latent function space? Defaults to False.
            n_samples (int): number of coarse grid points to sample for optimization estimate. Defaults to 1000.
            max_time (float, optional): Maximum time to spend optimizing. Defaults to None.
            weights (torch.Tensor, optional): Weights for the optimization. Defaults to None.

        Returns:
            Tuple[float, torch.Tensor]: Tuple containing the value of f
                nearest to queried y and the untransformed x position of this value.
        """
        val, loc = self._base_obj.inv_query(  # type: ignore
            y=y,
            locked_dims=locked_dims,
            probability_space=probability_space,
            n_samples=n_samples,
            max_time=max_time,
            weights=weights,
        )

        loc = self.transforms.untransform(loc)
        return val, loc

    def get_jnd(
        self,
        grid: Optional[Union[np.ndarray, torch.Tensor]] = None,
        cred_level: Optional[float] = None,
        intensity_dim: int = -1,
        confsamps: int = 500,
        method: str = "step",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Calculate the JND.

        Note that JND can have multiple plausible definitions
        outside of the linear case, so we provide options for how to compute it.
        For method="step", we report how far one needs to go over in stimulus
        space to move 1 unit up in latent space (this is a lot of people's
        conventional understanding of the JND).
        For method="taylor", we report the local derivative, which also maps to a
        1st-order Taylor expansion of the latent function. This is a formal
        generalization of JND as defined in Weber's law.
        Both definitions are equivalent for linear psychometric functions.

        Args:
            grid (torch.Tensor, optional): Untransformed mesh grid over which to find the JND.
                Defaults to a square grid of size as determined by aepsych.utils.dim_grid.
            cred_level (float, optional): Credible level for computing an interval.
                Defaults to None, computing no interval.
            intensity_dim (int): Dimension over which to compute the JND.
                Defaults to -1.
            confsamps (int): Number of posterior samples to use for
                computing the credible interval. Defaults to 500.
            method (str): "taylor" or "step" method (see docstring).
                Defaults to "step".

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: either the
                mean JND, or a median, lower, upper tuple of the JND posterior. All values
                are in the untransformed space.
        """
        jnds = self._base_obj.get_jnd(  # type: ignore
            grid=grid,
            cred_level=cred_level,
            intensity_dim=intensity_dim,
            confsamps=confsamps,
            method=method,
        )

        if isinstance(jnds, torch.Tensor):
            jnds = self.transforms.untransform(jnds)
        else:
            jnds = [self.transforms.untransform(jnd) for jnd in jnds]
            jnds = tuple(jnds)

        return jnds

    def p_below_threshold(
        self, x: torch.Tensor, f_thresh: torch.Tensor
    ) -> torch.Tensor:
        """Compute the probability that the latent function is below a threshold.

        Args:
            x (torch.Tensor): Points at which to evaluate the probability.
            f_thresh (torch.Tensor): Threshold value.

        Returns:
            torch.Tensor: Probability that the latent function is below the threshold.
        """
        x = self.transforms.transform(x)
        return self._base_obj.p_below_threshold(x, f_thresh)

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return a dictionary of the relevant options to initialize a model wrapped with
        a parameter transform

        Args:
            config (Config): Config to look for options in.
            name (str, optional): Strategy to find options for.
            options (Dict[str, Any], optional): Options to override from the config.

        Returns:
            Dict[str, Any]: A diciontary of options to initialize this class with.
        """
        if options is None:
            options = {}
            options["transforms"] = ParameterTransforms.from_config(config)
        else:
            # Check if there's a transform already if so save it to it persists over copying
            if "transforms" in options:
                transforms = options["transforms"]
            else:
                transforms = ParameterTransforms.from_config(config)

            options = deepcopy(options)
            options["transforms"] = transforms

        if name is None:
            raise ValueError("name of strategy must be set to initialize a model")
        else:
            model_cls = config.getobj(name, "model")

        # Transform config
        transformed_config = transform_options(config, options["transforms"])

        options["model"] = model_cls.from_config(transformed_config)

        return options


def transform_options(
    config: Config, transforms: Optional[ChainedInputTransform] = None
) -> Config:
    """Return a copy of the config with the options transformed.

    Args:
        config (Config): The config to be transformed.

    Returns:
        Config: A copy of the original config with relevant options transformed.
    """
    if transforms is None:
        transforms = ParameterTransforms.from_config(config)

    configClone = deepcopy(config)

    # Can't use self.sections() to avoid default section behavior
    for section, options in config.to_dict().items():
        for option, value in options.items():
            if option in _TRANSFORMABLE:
                value = ast.literal_eval(value)
                value = np.array(value, dtype=float)
                value = torch.tensor(value).to(torch.float64)

                if option in ["ub", "lb"]:
                    value = transforms.transform_bounds(value, bound=option)
                elif option == "points" and len(value.shape) == 3:
                    value = value.swapaxes(-2, -1)
                    value = transforms.transform(value)
                    value = value.swapaxes(-1, -2)
                elif option == "window":
                    # Get proportion of range covered in raw space
                    raw_lb = config.gettensor("common", "lb")
                    raw_ub = config.gettensor("common", "ub")
                    raw_range = raw_ub - raw_lb
                    window_prop = (value * 2) / raw_range

                    # Calculate transformed range
                    transformed_lb = transforms.transform_bounds(raw_lb, bound="lb")
                    transformed_ub = transforms.transform_bounds(raw_ub, bound="ub")
                    transformed_range = transformed_ub - transformed_lb

                    # Transformed window covers the same proportion of range
                    value = window_prop * transformed_range / 2
                else:
                    value = transforms.transform(value)

                def _arr_to_list(iter):
                    if hasattr(iter, "__iter__"):
                        iter = list(iter)
                        iter = [_arr_to_list(element) for element in iter]
                        return iter
                    return iter

                # Recursively turn back into str
                configClone[section][option] = str(_arr_to_list(value.numpy()))

    return configClone
