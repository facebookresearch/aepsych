#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc
import ast
import configparser
import json
import logging
import re
import warnings
from types import ModuleType
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

import botorch
import gpytorch
import numpy as np
import torch
from aepsych.version import __version__

_T = TypeVar("_T")
_ET = TypeVar("_ET")


class Config(configparser.ConfigParser):
    # names in these packages can be referred to by string name
    registered_names: ClassVar[Dict[str, object]] = {}

    def __init__(
        self,
        config_dict: Optional[Mapping[str, Any]] = None,
        config_fnames: Optional[Sequence[str]] = None,
        config_str: Optional[str] = None,
    ) -> None:
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
        section: str,
        conv: _T,
        option: str,
        *,
        raw: bool = False,
        vars: Optional[Dict[str, Any]] = None,
        fallback: _T = configparser._UNSET,
        **kwargs,
    ):
        """
        Override configparser to:
        1. Return from common if a section doesn't exist. This comes
        up any time we have a module fully configured from the
        common/default section.
        2. Pass extra **kwargs to the converter.


        Args:
            section (str): Section to get the option from.
            conv (_T): Converter to use.
            option (str): Option to get.
            raw (bool): Whether to return the raw value. Defaults to False.
            vars (Dict[str, Any], optional): Optional dictionary to use for interpolation. Defaults to None.
            fallback (_T): Value to return if the option is not found. Defaults to configparser._UNSET.

        Returns:
            _T: Converted value of the option.

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
    def to_dict(self, deduplicate: bool = True) -> Dict[str, Any]:
        """Convert the config into a dictionary.

        Args:
            deduplicate (bool): Whether to deduplicate the 'common' section. Defaults to True.

        Returns:
            dict: Dictionary representation of the config.
        """
        _dict: Dict[str, Any] = {}
        for section in self:
            _dict[section] = {}
            for setting in self[section]:
                if deduplicate and section != "common" and setting in self["common"]:
                    continue
                _dict[section][setting] = self[section][setting]
        return _dict

    def get_metadata(self, only_extra: bool = False) -> Dict[Any, Any]:
        """Return a dictionary of the metadata section.

        Args:
            only_extra (bool, optional): Only gather the extra metadata. Defaults to False.

        Returns:
            Dict[Any, Any]: a collection of the metadata stored in this conig.
        """
        configdict = self.to_dict()
        metadata = configdict["metadata"].copy()

        if only_extra:
            default_metadata = [
                "experiment_name",
                "experiment_description",
                "experiment_id",
                "participant_id",
            ]
            for name in default_metadata:
                metadata.pop(name, None)

        return metadata

    # Turn the metadata section into JSON.
    def jsonifyMetadata(self, only_extra: bool = False) -> str:
        """Return a json string of the metadata section.

        Args:
            only_extra (bool): Only jsonify the extra meta data.

        Returns:
            str: A json string representing the metadata dictionary or an empty string
                if there is no metadata to return.
        """
        metadata = self.get_metadata(only_extra)
        if len(metadata.keys()) == 0:
            return ""
        else:
            return json.dumps(metadata)

    # Turn the entire config into JSON format.
    def jsonifyAll(self) -> str:
        """Turn the entire config into JSON format.

        Returns:
            str: JSON representation of the entire config.
        """
        configdict = self.to_dict()
        return json.dumps(configdict)

    def update(
        self,
        config_dict: Optional[Mapping[str, str]] = None,
        config_fnames: Sequence[str] = None,
        config_str: str = None,
    ) -> None:
        """Update this object with a new configuration.

        Args:
            config_dict (Mapping[str, str], optional): Mapping to build configuration from.
                Keys are section names, values are dictionaries with keys and values that
                should be present in the section. Defaults to None.
            config_fnames (Sequence[str]): List of INI filenames to load
                configuration from. Defaults to None.
            config_str (str): String formatted as an INI file to load configuration
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

        if "parnames" in self["common"]:  # it's possible to pass no parnames
            try:
                par_names = self.getlist(
                    "common", "parnames", element_type=str, fallback=[]
                )
                lb = [None] * len(par_names)
                ub = [None] * len(par_names)
                for i, par_name in enumerate(par_names):
                    # Validate the parameter-specific block
                    self._check_param_settings(par_name)

                    lb[i] = self[par_name].get("lower_bound", fallback="0")
                    ub[i] = self[par_name].get("upper_bound", fallback="1")

                self["common"]["lb"] = f"[{', '.join(lb)}]"
                self["common"]["ub"] = f"[{', '.join(ub)}]"
            except ValueError:
                # Check if ub/lb exists in common
                if "ub" in self["common"] and "lb" in self["common"]:
                    logging.warning(
                        "Parameter-specific bounds are incomplete, falling back to ub/lb in [common]"
                    )
                else:
                    raise ValueError(
                        "Missing ub or lb in [common] with incomplete parameter-specific bounds, cannot fallback!"
                    )

        # Deprecation warning for "experiment" section
        if "experiment" in self:
            for i in self["experiment"]:
                self["common"][i] = self["experiment"][i]
            del self["experiment"]

    def _str_to_list(
        self, v: str, element_type: Callable[[_ET], _ET] = float
    ) -> List[_T]:
        """Convert a string to a list.

        Args:
            v (str): String to convert.
            element_type (Callable[[_ET], _ET]): Type of the elements in the list. Defaults to float.

        Returns:
            List[_T]: List of elements of type _T.
        """
        v = re.sub(r",]", "]", v)
        if re.search(r"^\[.*\]$", v, flags=re.DOTALL):
            if v == "[]":  # empty list
                return []
            else:
                return [element_type(i.strip()) for i in v[1:-1].split(",")]
        else:
            return [v.strip()]

    def _str_to_array(self, v: str) -> np.ndarray:
        """Convert a string to a numpy array.

        Args:
            v (str): String to convert.

        Returns:
            np.ndarray: Numpy array representation of the string.
        """
        v = ast.literal_eval(v)
        return np.array(v, dtype=float)

    def _str_to_tensor(self, v: str) -> torch.Tensor:
        """Convert a string to a torch tensor.

        Args:
            v (str): String to convert.

        Returns:
            torch.Tensor: Tensor representation of the string.
        """
        return torch.Tensor(self._str_to_array(v)).to(torch.float64)

    def _str_to_obj(self, v: str, fallback_type: _T = str, warn: bool = True) -> object:
        """Convert a string to an object.

        Args:
            v (str): String to convert.
            fallback_type (_T): Type to fallback to if the object is not found. Defaults to str.
            warn (bool): Whether to warn if the object is not found. Defaults to True.

        Returns:
            object: Object representation of the string.
        """
        try:
            return self.registered_names[v]
        except KeyError:
            if warn:
                warnings.warn(f'No known object "{v}"!')
            return fallback_type(v)

    def _check_param_settings(self, param_name: str) -> None:
        """Check parameter-specific blocks have the correct settings, raises a ValueError if not.

        Args:
            param_name (str): Parameter block to check.
        """
        # Check if the config block exists at all
        if param_name not in self:
            raise ValueError(f"Parameter {param_name} is missing its own config block.")

        param_block = self[param_name]

        # Checking if param_type is set
        if "par_type" not in param_block:
            raise ValueError(f"Parameter {param_name} is missing the par_type setting.")

        # Each parameter type has a different set of required settings
        if param_block["par_type"] == "continuous":
            # Check if bounds exist
            if "lower_bound" not in param_block:
                raise ValueError(
                    f"Parameter {param_name} is missing the lower_bound setting."
                )
            if "upper_bound" not in param_block:
                raise ValueError(
                    f"Parameter {param_name} is missing the upper_bound setting."
                )

        elif param_block["par_type"] == "integer":
            # Check if bounds exist and actaully integers
            if "lower_bound" not in param_block:
                raise ValueError(
                    f"Parameter {param_name} is missing the lower_bound setting."
                )
            if "upper_bound" not in param_block:
                raise ValueError(
                    f"Parameter {param_name} is missing the upper_bound setting."
                )

            try:
                if not (
                    self.getint(param_name, "lower_bound") % 1 == 0
                    and self.getint(param_name, "upper_bound") % 1 == 0
                ):
                    raise ParameterConfigError(
                        f"Parameter {param_name} has non-integer bounds."
                    )
            except ValueError:
                raise ParameterConfigError(
                    f"Parameter {param_name} has non-discrete bounds."
                )

        elif param_block["par_type"] == "binary":
            if "lower_bound" in param_block or "upper_bound" in param_block:
                raise ParameterConfigError(
                    f"Parameter {param_name} is binary and shouldn't have bounds."
                )

        elif param_block["par_type"] == "fixed":
            if "value" not in param_block:
                raise ParameterConfigError(
                    f"Parameter {param_name} is fixed and needs to have value set."
                )

        else:
            raise ParameterConfigError(
                f"Parameter {param_name} has an unsupported parameter type {param_block['par_type']}."
            )

    def __repr__(self) -> str:
        """Return a string representation of the config.

        Returns:
            str: String representation of the config.
        """
        return f"Config at {hex(id(self))}: \n {str(self)}"

    @classmethod
    def register_module(cls: _T, module: ModuleType) -> None:
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
    def register_object(cls: _T, obj: object) -> None:
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

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a section of the config.

        Args:
            section (str): Section to get.

        Returns:
            Dict[str, Any]: Dictionary representation of the section.
        """
        sec = {}
        for setting in self[section]:
            if section != "common" and setting in self["common"]:
                continue
            sec[setting] = self[section][setting]
        return sec

    def __str__(self):
        """Return a string representation of the config."""
        _str = ""
        for section in self:
            sec = self.get_section(section)
            _str += f"[{section}]\n"
            for setting in sec:
                _str += f"{setting} = {self[section][setting]}\n"
        return _str

    def convert_to_latest(self):
        """Converts the config to the latest version in place."""
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
        """Returns the version number of the config.

        Returns:
            str: Version number of the config.
        """
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


class ConfigurableMixin(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return a dictionary of the relevant options to initialize this class from the
        config, even if it is outside of the named section.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): Primary section to look for options for this class and
                the name to infer options from other sections in the config.
            options (Dict[str, Any], optional): Options to override from the config,
                defaults to None.


        Return:
            Dict[str, Any]: A dictionary of options to initialize this class.
        """

        raise NotImplementedError(
            f"get_config_options hasn't been defined for {cls.__name__}!"
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> "ConfigurableMixin":
        """
        Return a initialized instance of this class using the config and the name.

        Args:
            config (Config): Config to use to initialize this class.
            name (str, optional): Name of section to look in first for this class.
            options (Dict[str, Any], optional): Options to override from the config,
                defaults to None.

        Return:
            ConfigurableMixin: Initialized class based on config and name.
        """

        return cls(**cls.get_config_options(config, name, options))


Config.register_module(gpytorch.likelihoods)
Config.register_module(gpytorch.kernels)
Config.register_module(botorch.acquisition)
Config.register_module(botorch.acquisition.multi_objective)
Config.registered_names["None"] = None


class ParameterConfigError(Exception):
    pass
