#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import configparser
import pprint
import warnings
from types import ModuleType
from typing import Dict, TypeVar

import botorch
import gpytorch
import torch

_T = TypeVar("_T")


class Config(configparser.ConfigParser):

    # names in these packages can be referred to by string name
    registered_names: Dict[str, object] = {}

    def __init__(
        self, config_dict=None, config_fnames=None, config_list=None, config_str=None
    ):
        super().__init__(
            inline_comment_prefixes=("#"),
            empty_lines_in_values=False,
            default_section="common",
            interpolation=configparser.ExtendedInterpolation(),
            converters={
                "list": self.str_to_list,
                "tensor": self.str_to_tensor,
                "obj": self.str_to_obj,
            },
        )

        self.update(
            config_dict=config_dict,
            config_fnames=config_fnames,
            config_list=config_list,
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
        self, config_dict=None, config_fnames=None, config_list=None, config_str=None
    ):
        if config_dict is not None:
            self.read_dict(config_dict)

        if config_fnames is not None:
            self.read(config_fnames)

        if config_list is not None:
            self.read(config_list)

        if config_str is not None:
            self.read_string(config_str)

    def str_to_list(self, v, element_type=float):
        if v[0] == "[" and v[-1] == "]":
            if v == "[]":  # empty list
                return []
            else:
                return [element_type(i.strip()) for i in v[1:-1].split(",")]
        else:
            return [v.strip()]

    def str_to_tensor(self, v):
        return torch.Tensor(self.str_to_list(v))

    def str_to_obj(self, v, fallback_type=str, warn=True):

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
        cls.registered_names.update(
            {
                name: getattr(module, name)
                for name in module.__all__
                if not isinstance(getattr(module, name), ModuleType)
            }
        )

    @classmethod
    def register_object(cls: _T, obj: object):
        if obj.__name__ in cls.registered_names.keys():
            warnings.warn(
                f"Registering {obj.__name__} but already"
                + f"have {cls.registered_names[obj.__name__]}"
                + "registered under that name!"
            )
        cls.registered_names.update({obj.__name__: obj})


Config.register_module(gpytorch.kernels)
Config.register_module(botorch.acquisition)
