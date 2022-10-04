#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
import configparser
import json
import warnings
from types import ModuleType
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, TypeVar

import botorch
import gpytorch
import numpy as np
import torch

from aepsych.version import __version__

_T = TypeVar("_T")


class Config(configparser.ConfigParser):

    # names in these packages can be referred to by string name
    registered_names: ClassVar[Dict[str, object]] = {}

    def __init__(
        self,
        config_dict: Optional[Mapping[str, Any]] = None,
        config_fnames: Optional[Sequence[str]] = None,
        config_str: Optional[str] = None,
    ):
        """Initialize the AEPsych config object. This can be used to instantiate most
        objects in AEPsych by calling object.from_config(config).

        Args:
            config_dict (Mapping[str, str], optional): Mapping to build configuration from.
                Keys are section names, values are dictionaries with keys and values that
                should be present in the section. Defaults to None.
            config_fnames (Sequence[str], optional): List of INI filenames to load
                configuration from. Defaults to None.
            config_str (str, optional): String formatted as an INI file to load configuration
                from. Defaults to None.
        """
        super().__init__(
            inline_comment_prefixes=("#"),
            empty_lines_in_values=False,
            default_section="common",
            interpolation=configparser.ExtendedInterpolation(),
            converters={
                "list": self._str_to_list,
                "tensor": self._str_to_tensor,
                "obj": self._str_to_obj,
                "array": self._str_to_array,
            },
            allow_no_value=True,
        )

        self.update(
            config_dict=config_dict,
            config_fnames=config_fnames,
            config_str=config_str,
        )

    def _get(
        self,
        section,
        conv,
        option,
        *,
        raw=False,
        vars=None,
        fallback=configparser._UNSET,
        **kwargs,
    ):

        """
        Override configparser to:
        1. Return from common if a section doesn't exist. This comes
        up any time we have a module fully configured from the
        common/default section.
        2. Pass extra **kwargs to the converter.
        """
        try:
            return conv(
                self.get(
                    section=section,
                    option=option,
                    raw=raw,
                    vars=vars,
                    fallback=fallback,
                ),
                **kwargs,
            )
        except configparser.NoSectionError:
            return conv(
                self.get(
                    section="common",
                    option=option,
                    raw=raw,
                    vars=vars,
                    fallback=fallback,
                ),
                **kwargs,
            )

    # Convert config into a dictionary (eliminate duplicates from defaulted 'common' section.)
    def to_dict(self, deduplicate=True):
        _dict = {}
        for section in self:
            _dict[section] = {}
            for setting in self[section]:
                if deduplicate and section != "common" and setting in self["common"]:
                    continue
                _dict[section][setting] = self[section][setting]
        return _dict

    # Turn the metadata section into JSON.
    def jsonifyMetadata(self):
        configdict = self.to_dict()
        return json.dumps(configdict["metadata"])

    # Turn the entire config into JSON format.
    def jsonifyAll(self):
        configdict = self.to_dict()
        return json.dumps(configdict)

    def update(
        self,
        config_dict: Mapping[str, str] = None,
        config_fnames: Sequence[str] = None,
        config_str: str = None,
    ):
        """Update this object with a new configuration.

        Args:
            config_dict (Mapping[str, str], optional): Mapping to build configuration from.
                Keys are section names, values are dictionaries with keys and values that
                should be present in the section. Defaults to None.
            config_fnames (Sequence[str], optional): List of INI filenames to load
                configuration from. Defaults to None.
            config_str (str, optional): String formatted as an INI file to load configuration
                from. Defaults to None.
        """
        if config_dict is not None:
            self.read_dict(config_dict)

        if config_fnames is not None:
            read_ok = self.read(config_fnames)
            if len(read_ok) < 1:
                raise FileNotFoundError

        if config_str is not None:
            self.read_string(config_str)

        # Deprecation warning for "experiment" section
        if "experiment" in self:
            for i in self["experiment"]:
                self["common"][i] = self["experiment"][i]
            del self["experiment"]

    def _str_to_list(self, v: str, element_type: _T = float) -> List[_T]:
        if v[0] == "[" and v[-1] == "]":
            if v == "[]":  # empty list
                return []
            else:
                return [element_type(i.strip()) for i in v[1:-1].split(",")]
        else:
            return [v.strip()]

    def _str_to_array(self, v: str) -> np.ndarray:
        v = ast.literal_eval(v)
        return np.array(v, dtype=float)

    def _str_to_tensor(self, v: str) -> torch.Tensor:
        return torch.Tensor(self._str_to_list(v))

    def _str_to_obj(self, v: str, fallback_type: _T = str, warn: bool = True) -> object:

        try:
            return self.registered_names[v]
        except KeyError:
            if warn:
                warnings.warn(f'No known object "{v}"!')
            return fallback_type(v)

    def __repr__(self):
        return f"Config at {hex(id(self))}: \n {str(self)}"

    @classmethod
    def register_module(cls: _T, module: ModuleType):
        """Register a module with Config so that objects in it can
           be referred to by their string name in config files.

        Args:
            module (ModuleType): Module to register.
        """
        cls.registered_names.update(
            {
                name: getattr(module, name)
                for name in module.__all__
                if not isinstance(getattr(module, name), ModuleType)
            }
        )

    @classmethod
    def register_object(cls: _T, obj: object):
        """Register an object with Config so that it can be
            referred to by its string name in config files.

        Args:
            obj (object): Object to register.
        """
        if obj.__name__ in cls.registered_names.keys():
            warnings.warn(
                f"Registering {obj.__name__} but already"
                + f"have {cls.registered_names[obj.__name__]}"
                + "registered under that name!"
            )
        cls.registered_names.update({obj.__name__: obj})

    def get_section(self, section):
        sec = {}
        for setting in self[section]:
            if section != "common" and setting in self["common"]:
                continue
            sec[setting] = self[section][setting]
        return sec

    def __str__(self):
        _str = ""
        for section in self:
            sec = self.get_section(section)
            _str += f"[{section}]\n"
            for setting in sec:
                _str += f"{setting} = {self[section][setting]}\n"
        return _str

    def convert_to_latest(self):
        self.convert(self.version, __version__)

    def convert(self, from_version: str, to_version: str) -> None:
        """Converts a config from an older version to a newer version.

        Args:
            from_version (str): The version of the config to be converted.
            to_version (str): The version the config should be converted to.
        """

        if from_version == "0.0":
            self["common"]["strategy_names"] = "[init_strat, opt_strat]"

            if "experiment" in self:
                for i in self["experiment"]:
                    self["common"][i] = self["experiment"][i]

            bridge = self["common"]["modelbridge_cls"]
            n_sobol = self["SobolStrategy"]["n_trials"]
            n_opt = self["ModelWrapperStrategy"]["n_trials"]

            if bridge == "PairwiseProbitModelbridge":
                self["init_strat"] = {
                    "generator": "PairwiseSobolGenerator",
                    "min_asks": n_sobol,
                }
                self["opt_strat"] = {
                    "generator": "PairwiseOptimizeAcqfGenerator",
                    "model": "PairwiseProbitModel",
                    "min_asks": n_opt,
                }
                if "PairwiseProbitModelbridge" in self:
                    self["PairwiseOptimizeAcqfGenerator"] = self[
                        "PairwiseProbitModelbridge"
                    ]
                if "PairwiseGP" in self:
                    self["PairwiseProbitModel"] = self["PairwiseGP"]

            elif bridge == "MonotonicSingleProbitModelbridge":
                self["init_strat"] = {
                    "generator": "SobolGenerator",
                    "min_asks": n_sobol,
                }
                self["opt_strat"] = {
                    "generator": "MonotonicRejectionGenerator",
                    "model": "MonotonicRejectionGP",
                    "min_asks": n_opt,
                }
                if "MonotonicSingleProbitModelbridge" in self:
                    self["MonotonicRejectionGenerator"] = self[
                        "MonotonicSingleProbitModelbridge"
                    ]

            elif bridge == "SingleProbitModelbridge":
                self["init_strat"] = {
                    "generator": "SobolGenerator",
                    "min_asks": n_sobol,
                }
                self["opt_strat"] = {
                    "generator": "OptimizeAcqfGenerator",
                    "model": "GPClassificationModel",
                    "min_asks": n_opt,
                }
                if "SingleProbitModelbridge" in self:
                    self["OptimizeAcqfGenerator"] = self["SingleProbitModelbridge"]

            else:
                raise NotImplementedError(
                    f"Refactor for {bridge} has not been implemented!"
                )

            if "ModelWrapperStrategy" in self:
                if "refit_every" in self["ModelWrapperStrategy"]:
                    self["opt_strat"]["refit_every"] = self["ModelWrapperStrategy"][
                        "refit_every"
                    ]

            del self["common"]["model"]

        if to_version == __version__:
            if self["common"]["outcome_type"] == "single_probit":
                self["common"]["stimuli_per_trial"] = "1"
                self["common"]["outcome_types"] = "[binary]"

            if self["common"]["outcome_type"] == "single_continuous":
                self["common"]["stimuli_per_trial"] = "1"
                self["common"]["outcome_types"] = "[continuous]"

            if self["common"]["outcome_type"] == "pairwise_probit":
                self["common"]["stimuli_per_trial"] = "2"
                self["common"]["outcome_types"] = "[binary]"

            del self["common"]["outcome_type"]

    @property
    def version(self) -> str:
        """Returns the version number of the config."""
        # TODO: implement an explicit versioning system

        # Try to infer the version
        if "stimuli_per_trial" in self["common"] and "outcome_types" in self["common"]:
            return __version__

        if "common" in self and "strategy_names" in self["common"]:
            return "0.1"

        elif (
            "SobolStrategy" in self
            or "ModelWrapperStrategy" in self
            or "EpsilonGreedyModelWrapperStrategy" in self
        ):
            return "0.0"

        else:
            raise RuntimeError("Unrecognized config format!")


Config.register_module(gpytorch.kernels)
Config.register_module(botorch.acquisition)
Config.registered_names["None"] = None
