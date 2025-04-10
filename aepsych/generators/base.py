#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc
import re
from inspect import _empty, signature
from typing import Any, Generic, Protocol, runtime_checkable, TypeVar

import torch
from aepsych.config import Config, ConfigurableMixin
from aepsych.models.base import AEPsychModelMixin
from aepsych.utils_logging import logger
from botorch.acquisition import (
    AcquisitionFunction,
    LogNoisyExpectedImprovement,
    NoisyExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption

AEPsychModelType = TypeVar("AEPsychModelType", bound=AEPsychModelMixin)


@runtime_checkable
class AcqArgProtocol(Protocol):
    @classmethod
    def from_config(cls, config: Config) -> Any:
        pass


class AEPsychGenerator(ConfigurableMixin, abc.ABC, Generic[AEPsychModelType]):
    """Abstract base class for generators, which are responsible for generating new points to sample."""

    _requires_model = True
    stimuli_per_trial = 1
    max_asks: int | None = None
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
        fixed_features: dict[int, float] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        pass


class AcqfGenerator(AEPsychGenerator):
    """Base class for generators that evaluate acquisition functions."""

    _requires_model = True
    baseline_requiring_acqfs = [
        qNoisyExpectedImprovement,
        NoisyExpectedImprovement,
        qLogNoisyExpectedImprovement,
        LogNoisyExpectedImprovement,
    ]
    acqf: AcquisitionFunction
    acqf_kwargs: dict[str, Any]

    def __init__(
        self,
        acqf: AcquisitionFunction,
        acqf_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.acqf = acqf
        if acqf_kwargs is None:
            acqf_kwargs = {}
        self.acqf_kwargs = acqf_kwargs

    @classmethod
    def _get_acqf_options(
        cls, acqf: AcquisitionFunction, config: Config
    ) -> dict[str, Any]:
        """Get the extra arguments for the acquisition function from the config.

        Args:
            acqf (AcquisitionFunction): The acquisition function to get arguments for.
            config (Config): The configuration object.

        Returns:
            dict[str, Any]: The extra arguments for the acquisition function.
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
                        logger.debug(
                            f"{acqf_name} requires the {key} option but we could not find it."
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

    def _instantiate_acquisition_fn(
        self, model: AEPsychModelMixin
    ) -> AcquisitionFunction:
        """
        Instantiates the acquisition function with the specified model and additional arguments.

        Args:
            model (AEPsychModelMixin): The model to use with the acquisition function.

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
            if model.train_inputs is None:
                raise ValueError(f"model needs data as a baseline for {self.acqf}")
            return self.acqf(model, model.train_inputs[0], **self.acqf_kwargs)
        else:
            return self.acqf(model=model, **self.acqf_kwargs)

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get configuration options for the generator.

        Args:
            config (Config): Configuration object.
            name (str, optional): Name of the generator, defaults to None. Ignored.
            options (dict[str, Any], optional): Additional options, defaults to None.

        Returns:
            dict[str, Any]: Configuration options for the generator.
        """
        name = name or cls.__name__
        options = super().get_config_options(config=config, name=name, options=options)

        options.update({"acqf_kwargs": cls._get_acqf_options(options["acqf"], config)})

        return options
