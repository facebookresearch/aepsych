#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import configparser
import pprint
import warnings
from types import ModuleType
from typing import Dict, TypeVar, Mapping, Sequence, List

import botorch
import gpytorch
import torch

_T = TypeVar("_T")


class Config(configparser.ConfigParser):

    # names in these packages can be referred to by string name
    registered_names: Dict[str, object] = {}

    def __init__(
        self,
        config_dict: Mapping[str, str] = None,
        config_fnames: Sequence[str] = None,
        config_str: str = None,
    ):
        """Initialize the AEPsych config object. This can be used to instantiate most
        objects in AEPsych by calling object.from_config(config).

        TODO write a tutorial on writing configs.

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
            },
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

        # Deprecation warining for "experiment" section
        if "experiment" in self:
            for i in self["experiment"]:
                self["common"][i] = self["experiment"][i]
            del self["experiment"]
            warnings.warn(
                'The "experiment" section is being deprecated from configs. Please put everything in the "experiment" section in the "common" section instead.',
                DeprecationWarning,
            )

    def _str_to_list(self, v: str, element_type: _T = float) -> List[_T]:
        if v[0] == "[" and v[-1] == "]":
            if v == "[]":  # empty list
                return []
            else:
                return [element_type(i.strip()) for i in v[1:-1].split(",")]
        else:
            return [v.strip()]

    def _str_to_tensor(self, v: str) -> torch.Tensor:
        return torch.Tensor(self._str_to_list(v))

    def _str_to_obj(self, v: str, fallback_type: _T = str, warn: bool = True) -> object:

        try:
            return self.registered_names[v]
        except KeyError:
            if warn:
                warnings.warn(f'No known object "{v}"!')
            return fallback_type(v)

    def __str__(self):
        defaults = {k: v for k, v in self.items("common")}
        nondefaults = {
            sec: {k: v for k, v in self[sec].items() if k not in defaults.keys()}
            for sec in self.sections()
        }
        nondefaults["common"] = defaults

        return pprint.pformat(nondefaults)

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


Config.register_module(gpytorch.kernels)
Config.register_module(botorch.acquisition)
