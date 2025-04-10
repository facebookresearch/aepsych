#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc
import ast
import configparser
import inspect
import json
import re
import typing
import warnings
from types import ModuleType, NoneType, UnionType
from typing import Any, Callable, ClassVar, Mapping, Sequence, TypeVar

import botorch
import gpytorch
import numpy as np
import torch
from aepsych.utils_logging import logger

_T = TypeVar("_T")
_ET = TypeVar("_ET")

DEPRECATED_OBJS = [
    "MonotonicRejectionGenerator",
    "MonotonicMCPosteriorVariance",
    "MonotonicBernoulliMCMutualInformation",
    "MonotonicMCLSE",
    "MonotonicRejectionGP",
    "monotonic_mean_covar_factory",
]


class Config(configparser.ConfigParser):
    # names in these packages can be referred to by string name
    registered_names: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        config_dict: Mapping[str, Any] | None = None,
        config_fnames: Sequence[str] | None = None,
        config_str: str | None = None,
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
        vars: dict[str, Any] | None = None,
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
            vars (dict[str, Any], optional): Optional dictionary to use for interpolation. Defaults to None.
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
    def to_dict(self, deduplicate: bool = True) -> dict[str, Any]:
        """Convert the config into a dictionary.

        Args:
            deduplicate (bool): Whether to deduplicate the 'common' section. Defaults to True.

        Returns:
            dict: Dictionary representation of the config.
        """
        _dict: dict[str, Any] = {}
        for section in self:
            _dict[section] = {}
            for setting in self[section]:
                if deduplicate and section != "common" and setting in self["common"]:
                    continue
                _dict[section][setting] = self[section][setting]
        return _dict

    def get_metadata(self, only_extra: bool = False) -> dict[Any, Any]:
        """Return a dictionary of the metadata section.

        Args:
            only_extra (bool, optional): Only gather the extra metadata. Defaults to False.

        Returns:
            dict[Any, Any]: a collection of the metadata stored in this conig.
        """
        configdict = self.to_dict()
        metadata = configdict.get("metadata", {}).copy()

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
        config_dict: Mapping[str, str] | None = None,
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
                    logger.warning(
                        "Parameter-specific bounds are incomplete, falling back to ub/lb in [common]"
                    )
                else:
                    raise ValueError(
                        "Missing ub or lb in [common] with incomplete parameter-specific bounds, cannot fallback!"
                    )

    def _str_to_list(
        self, v: str, element_type: Callable[[_ET], _ET] = float
    ) -> list[_T]:
        """Convert a string to a list.

        Args:
            v (str): String to convert.
            element_type (Callable[[_ET], _ET]): Type of the elements in the list. Defaults to float.

        Returns:
            list[_T]: List of elements of type _T.
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
            # Check if this is an alias
            if v in self:
                # Return alias class with a new name
                class_ = self.getobj(v, "class")
                alias_class = type(
                    class_.__name__, (class_,), {}
                )  # Creates a deepcopy of the class
                alias_class.__name__ = v
                return alias_class

            if warn:
                if v in DEPRECATED_OBJS:
                    raise TypeError(
                        f"Object {v} is deprecated and no longer supported!"
                    )
                else:
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

    def get_section(self, section: str) -> dict[str, Any]:
        """Get a section of the config.

        Args:
            section (str): Section to get.

        Returns:
            dict[str, Any]: Dictionary representation of the section.
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


class ConfigurableMixin(abc.ABC):
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
            name = cls.__name__

        def _sort_types(annotations):
            # Rebuild the annotations, prefering float, int, string, then the rest
            reordered = []
            if float in annotations:
                reordered += [float]

            if int in annotations:
                reordered += [int]

            if str in annotations:
                reordered += [str]

            reordered += [elem for elem in annotations if elem not in [float, int, str]]
            return tuple(reordered)

        args = inspect.signature(cls, eval_str=True).parameters

        options = options or {}
        for key, signature in args.items():
            if signature.kind in [
                inspect.Parameter.KEYWORD_ONLY,  # Ignore *
                inspect.Parameter.VAR_POSITIONAL,  # Ignore *args
                inspect.Parameter.VAR_KEYWORD,  # Ignore **kwargs
            ]:
                continue
            # Used as fallback
            value = signature.default

            if typing.get_origin(signature.annotation) in [
                typing.Union,
                UnionType,
            ]:  # Includes Optional
                annotations = typing.get_args(signature.annotation)
                annotations = _sort_types(annotations)

            else:
                annotations = (signature.annotation,)

            for annotation in annotations:
                try:
                    # Tensor
                    if annotation is torch.Tensor:
                        value = config.gettensor(name, key)

                    # Numpy array
                    elif annotation is np.ndarray:
                        value = config.getarray(name, key)

                    # Default list
                    elif annotation is list:
                        try:
                            value = config.getlist(name, key, element_type=float)
                        except ValueError:
                            value = config.getlist(name, key, element_type=str)

                    # Generic List[...]
                    elif typing.get_origin(annotation) is list:
                        element_types = typing.get_args(annotation)
                        element_type = _sort_types(element_types)[0]

                        if isinstance(element_type, abc.ABCMeta):
                            break

                        value = config.getlist(
                            name,
                            key,
                            element_type=element_type,
                        )

                    # String
                    elif annotation is str:
                        value = config.get(name, key)

                    # Int
                    elif annotation is int:
                        value = config.getint(name, key)

                    # Float
                    elif annotation is float:
                        value = config.getfloat(name, key)

                    # Bool
                    elif annotation is bool:
                        value = config.getboolean(name, key)

                    # Object
                    elif inspect.isclass(annotation):
                        object_cls = config.getobj(name, key)
                        if hasattr(object_cls, "from_config"):
                            value = object_cls.from_config(config, object_cls.__name__)
                        else:
                            value = object_cls

                    # Callable
                    elif annotation is Callable:
                        value = config.getobj(name, key)

                    # None type
                    elif annotation is NoneType:
                        value = None

                    # We essentially keep trying until we succeed
                    break
                except (ValueError, configparser.NoOptionError):
                    pass

            if key not in options and value is not inspect._empty:
                options[key] = value

        return options

    @classmethod
    def from_config(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> "ConfigurableMixin":
        """
        Return a initialized instance of this class using the config and the name.

        Args:
            config (Config): Config to use to initialize this class.
            name (str, optional): Name of section to look in first for this class.
            options (dict[str, Any], optional): Options to override from the config,
                defaults to None.

        Return:
            ConfigurableMixin: Initialized class based on config and name.
        """

        return cls(**cls.get_config_options(config=config, name=name, options=options))


Config.register_module(gpytorch.likelihoods)
Config.register_module(gpytorch.kernels)
Config.register_module(botorch.acquisition)
Config.register_module(botorch.acquisition.multi_objective)
Config.registered_names["None"] = None


class ParameterConfigError(Exception):
    pass
