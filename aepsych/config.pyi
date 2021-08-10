#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import configparser
import pprint
import warnings
from types import ModuleType
from typing import Dict, Mapping, Optional, Any, overload, TypeVar, Union

import botorch
import gpytorch
import torch

_T = TypeVar("_T")

class Config(configparser.ConfigParser):
    @overload
    def gettensor(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Optional[Mapping[str, str]] = ...,
    ) -> torch.Tensor: ...
    @overload
    def gettensor(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Optional[Mapping[str, str]] = ...,
        fallback: _T = ...,
    ) -> Union[torch.Tensor, _T]: ...
    @overload
    def getobj(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Optional[Mapping[str, str]] = ...,
    ) -> Any: ...
    @overload
    def getobj(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Optional[Mapping[str, str]] = ...,
        fallback: _T = ...,
    ) -> Union[Any, _T]: ...
    @classmethod
    def register_module(cls: _T, module): ...
