#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc
import re
import warnings
from inspect import _empty, signature
from typing import Any, Dict, Generic, Optional, Protocol, runtime_checkable, TypeVar

import torch
from aepsych.config import Config, ConfigurableMixin
from aepsych.models.base import AEPsychMixin
from botorch.acquisition import (
    AcquisitionFunction,
    LogNoisyExpectedImprovement,
    NoisyExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption

from ..models.base import ModelProtocol

AEPsychModelType = TypeVar("AEPsychModelType", bound=AEPsychMixin)


@runtime_checkable
class AcqArgProtocol(Protocol):
    @classmethod
    def from_config(cls, config: Config) -> Any:
        pass


class AEPsychGenerator(abc.ABC, Generic[AEPsychModelType]):
    """Abstract base class for generators, which are responsible for generating new points to sample."""

    _requires_model = True
    stimuli_per_trial = 1
    max_asks: Optional[int] = None
    dim: int

    def __init__(
        self,
    ) -> None:
        pass

    @abc.abstractmethod
    def gen(
        self,
        num_points: int,
        model: AEPsychModelType,
        fixed_features: Optional[Dict[int, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        pass


class AcqfGenerator(AEPsychGenerator, ConfigurableMixin):
    """Base class for generators that evaluate acquisition functions."""

    _requires_model = True
    baseline_requiring_acqfs = [
        qNoisyExpectedImprovement,
        NoisyExpectedImprovement,
        qLogNoisyExpectedImprovement,
        LogNoisyExpectedImprovement,
    ]
    acqf: AcquisitionFunction
    acqf_kwargs: Dict[str, Any]

    def __init__(
        self,
        acqf: AcquisitionFunction,
        acqf_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.acqf = acqf
        if acqf_kwargs is None:
            acqf_kwargs = {}
        self.acqf_kwargs = acqf_kwargs

    @classmethod
    def _get_acqf_options(
        cls, acqf: AcquisitionFunction, config: Config
    ) -> Dict[str, Any]:
        """Get the extra arguments for the acquisition function from the config.

        Args:
            acqf (AcquisitionFunction): The acquisition function to get arguments for.
            config (Config): The configuration object.

        Returns:
            Dict[str, Any]: The extra arguments for the acquisition function.
        """

        if acqf is not None:
            acqf_name = acqf.__name__

            # model is not an extra arg, it's a default arg
            acqf_kwargs = signature(acqf).parameters
            acqf_args_expected = [i for i in list(acqf_kwargs.keys()) if i != "model"]

            # this is still very ugly
            extra_acqf_args = {}
            if acqf_name in config:
                full_section = config[acqf_name]
                for k in acqf_args_expected:
                    # if this thing is configured
                    if k in full_section.keys():
                        v = config.get(acqf_name, k)
                        # if it's an object make it an object
                        if full_section[k] in Config.registered_names.keys():
                            extra_acqf_args[k] = config.getobj(acqf_name, k)
                        elif re.search(
                            r"^\[.*\]$", v, flags=re.DOTALL
                        ):  # use regex to check if the value is a list
                            extra_acqf_args[k] = config._str_to_list(v)  # type: ignore
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

            # Final checks, bandaid
            for key, value in acqf_kwargs.items():
                if key == "model":  # Model is handled separately
                    continue
                if value.default == _empty and key not in extra_acqf_args:
                    if key not in config["common"]:
                        # HACK: Not actually sure why some required args can be missing
                        warnings.warn(
                            f"{acqf_name} requires the {key} option but we could not find it.",
                            UserWarning,
                        )
                        continue

                    # A required parameter is missing! Look for it in common
                    config_str = config.get("common", key)

                    # An object that we know about
                    if config_str in Config.registered_names.keys():
                        extra_acqf_args[key] = config.getobj("common", key)

                    # Some sequence
                    if "[" in config_str and "]" in config_str:
                        # Try to turn it into a tensor or fallback to list
                        try:
                            extra_acqf_args[key] = config.gettensor("common", key)
                        except ValueError:
                            extra_acqf_args[key] = config.getlist("common", key)

        else:
            extra_acqf_args = {}

        return extra_acqf_args

    def _instantiate_acquisition_fn(self, model: ModelProtocol) -> AcquisitionFunction:
        """
        Instantiates the acquisition function with the specified model and additional arguments.

        Args:
            model (ModelProtocol): The model to use with the acquisition function.

        Returns:
            AcquisitionFunction: Configured acquisition function.
        """
        if self.acqf == AnalyticExpectedUtilityOfBestOption:
            return self.acqf(pref_model=model)

        if hasattr(model, "device"):
            if "lb" in self.acqf_kwargs:
                if not isinstance(self.acqf_kwargs["lb"], torch.Tensor):
                    self.acqf_kwargs["lb"] = torch.tensor(self.acqf_kwargs["lb"])

                self.acqf_kwargs["lb"] = self.acqf_kwargs["lb"].to(model.device)

            if "ub" in self.acqf_kwargs:
                if not isinstance(self.acqf_kwargs["ub"], torch.Tensor):
                    self.acqf_kwargs["ub"] = torch.tensor(self.acqf_kwargs["ub"])

                self.acqf_kwargs["ub"] = self.acqf_kwargs["ub"].to(model.device)

        if self.acqf in self.baseline_requiring_acqfs:
            return self.acqf(model, model.train_inputs[0], **self.acqf_kwargs)
        else:
            return self.acqf(model=model, **self.acqf_kwargs)

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get configuration options for the generator.

        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the generator, defaults to None. Ignored.
            options (Dict[str, Any], optional): Additional options, defaults to None.

        Returns:
            Dict[str, Any]: Configuration options for the generator.
        """
        options = options or {}
        classname = cls.__name__
        acqf = config.getobj(classname, "acqf", fallback=None)
        extra_acqf_args = cls._get_acqf_options(acqf, config)
        options.update(
            {
                "acqf": acqf,
                "acqf_kwargs": extra_acqf_args,
            }
        )

        return options
