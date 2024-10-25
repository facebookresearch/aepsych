#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ast
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, List, Literal, Optional, Type

import numpy as np
import torch
from aepsych.config import Config
from aepsych.generators import SobolGenerator
from aepsych.generators.base import AEPsychGenerator
from aepsych.models.base import AEPsychMixin, ModelProtocol
from aepsych.transforms.log10 import Log10, Log10Plus
from botorch.acquisition import AcquisitionFunction
from botorch.models.transforms.input import ChainedInputTransform
from botorch.posteriors import Posterior
from torch import Tensor

_TRANSFORMABLE = [
    "lb",
    "ub",
    "points",
    "window",
]


class ParameterTransforms(ChainedInputTransform):
    """
    Holds set of transformations to be applied to parameters. The ParameterTransform
    objects can be used by themselves to transform values or can be passed to Generator
    or Model wrappers to consistently transform parameters. ParameterTransforms can
    transform values into transformed space and also untransform values from transformed
    space back into raw space.
    """

    @classmethod
    def from_config(cls, config: Config):
        parnames: List[str] = config.getlist("common", "parnames", element_type=str)
        transformDict = {}
        for i, par in enumerate(parnames):
            if config.getboolean(par, "log_scale", fallback=False):
                lb = config.getfloat(par, "lower_bound")
                if lb < 0.0:
                    raise ValueError(
                        f"Cannot log-scale values below 0. {par} lower bound set to {lb}"
                    )
                elif lb < 1.0:
                    transformDict[f"{par}_logPlus"] = Log10Plus(indices=[i])
                else:
                    transformDict[f"{par}_log"] = Log10(indices=[i])
        return cls(**transformDict)


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

    @classmethod
    @abstractmethod
    def from_config(cls, name: str, config: Config):
        pass


class GeneratorWrapper(ParameterTransformWrapper):
    _base_obj: AEPsychGenerator

    def __init__(
        self,
        generator: Type | AEPsychGenerator,
        transforms: ChainedInputTransform = ChainedInputTransform(**{}),
        **kwargs,
    ):
        f"""
        Wraps a Generator with parameter transforms. This will transform any relevant
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
        """
        # Figure out what we need to do with generator
        if isinstance(generator, type):
            if "lb" in kwargs:
                kwargs["lb"] = transforms.transform(kwargs["lb"].float())
            if "ub" in kwargs:
                kwargs["ub"] = transforms.transform(kwargs["ub"].float())
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

    def gen(self, num_points: int, model: Optional[AEPsychMixin] = None) -> Tensor:
        x = self._base_obj.gen(num_points, model)
        return self.transforms.untransform(x)

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
    def from_config(
        cls,
        name: str,
        config: Config,
    ):
        gen_cls = config.getobj(name, "generator", fallback=SobolGenerator)
        transforms = ParameterTransforms.from_config(config)

        # We need transformed values from config but we don't want to edit config
        transformed_config = transform_options(config)

        gen = gen_cls.from_config(transformed_config)

        return cls(gen, transforms)

    def _get_acqf_options(self, acqf: AcquisitionFunction, config: Config):
        return self._base_obj._get_acqf_options(acqf, config)


class ModelWrapper(ParameterTransformWrapper):
    _base_obj: ModelProtocol

    def __init__(
        self,
        model: Type | ModelProtocol,
        transforms: ChainedInputTransform = ChainedInputTransform(**{}),
        **kwargs,
    ):
        f"""
        Wraps a Model with parameter transforms. This will transform any relevant
        model arguments (e.g., bounds) and model data (e.g., training data, x) to be
        transformed into the transformed space. The wrapper surfaces the API of the
        raw model such that the wrapper can be used like a raw model.

        Bounds are returned in the transformed space, this is necessary to handle
        parameters that would not have sensible raw parameter space. If bounds are
        manually set (e.g., `Wrapper(**kwargs).lb = lb)`, ensure that they are
        correctly transformed and in a correctly shaped Tensor. If the bounds are
        being set in init (e.g., `Wrapper(Type, lb=lb, ub=ub)`, `lb` and `ub`
        should be in the raw parameter space.

        Args:
            model (Type | ModelProtocol): Model to wrap, this could either be a 
                completely initialized model or just the model class. An initialized
                model is expected to have been initialized in the transformed
                parameter space (i.e., bounds are transformed). If a model class is 
                passed, **kwargs will be used to initialize the model. Note that the 
                bounds in this case are expected to be in raw parameter space, thus the 
                transforms are applied to it.
            transforms (ChainedInputTransform, optional): A set of transforms to apply
                to parameters of this model. If no transforms are passed, it will 
                default to an identity transform.
        """
        # Alternative instantiation method for analysis (and not live)
        if isinstance(model, type):
            if "lb" in kwargs:
                kwargs["lb"] = transforms.transform(kwargs["lb"].float())
            if "ub" in kwargs:
                kwargs["ub"] = transforms.transform(kwargs["ub"].float())
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

    def predict(self, x: Tensor, **kwargs) -> Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        x = self.transforms.transform(x)
        return self._base_obj.predict(x, **kwargs)

    def predict_probability(self, x: Tensor, **kwargs) -> Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        x = self.transforms.transform(x)
        return self._base_obj.predict_probability(x, **kwargs)

    def sample(self, x: Tensor, num_samples: int) -> Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        x = self.transforms.transform(x)
        return self._base_obj.sample(x, num_samples)

    def dim_grid(self, gridsize: int = 30) -> Tensor:
        grid = self._base_obj.dim_grid(gridsize)
        return self.transforms.untransform(grid)

    def posterior(self, X: Tensor, **kwargs) -> Posterior:
        # This ensures X is a tensor with the right shape
        X = Tensor(X)
        return self._base_obj.posterior(X=X, **kwargs)

    def fit(self, train_x: Tensor, train_y: Tensor, **kwargs: Any) -> None:
        if len(train_x.shape) == 1:
            train_x = train_x.unsqueeze(-1)
        train_x = self.transforms.transform(train_x)
        self._base_obj.fit(train_x, train_y, **kwargs)

    def update(self, train_x: Tensor, train_y: Tensor, **kwargs: Any) -> None:
        if len(train_x.shape) == 1:
            train_x = train_x.unsqueeze(-1)
        train_x = self.transforms.transform(train_x)
        self._base_obj.update(train_x, train_y, **kwargs)

    def p_below_threshold(self, x: Tensor, f_thresh: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        x = self.transforms.transform(x)
        return self._base_obj.p_below_threshold(x, f_thresh)

    @classmethod
    def from_config(
        cls,
        name: str,
        config: Config,
    ):
        # We don't always have models
        model_cls = config.getobj(name, "model", fallback=None)
        if model_cls is None:
            return None

        transforms = ParameterTransforms.from_config(config)

        # Need transformed values
        transformed_config = transform_options(config)

        model = model_cls.from_config(transformed_config)

        return cls(model, transforms)


def transform_options(config: Config) -> Config:
    """
    Return a copy of the config with the options transformed. The config
    """
    transforms = ParameterTransforms.from_config(config)

    configClone = deepcopy(config)

    # Can't use self.sections() to avoid default section behavior
    for section, options in config.to_dict().items():
        for option, value in options.items():
            if option in _TRANSFORMABLE:
                value = ast.literal_eval(value)
                value = np.array(value, dtype=float)
                value = torch.tensor(value).to(torch.float64)

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
