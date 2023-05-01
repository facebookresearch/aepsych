#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import configparser
from typing import Any, ClassVar, Dict, List, Mapping, Optional, TypeVar, Union

import numpy as np
import torch

_T = TypeVar("_T")
_ET = TypeVar("_ET")

"""
This whole thing exists so that mypy is happy with the very
arcane way that ConfigParser handles type conversions.
"""

class Config(configparser.ConfigParser):
    registered_names: ClassVar[Dict[str, object]]
    def gettensor(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Optional[Mapping[str, str]] = ...,
        fallback: _T = ...,
    ) -> Union[torch.Tensor, _T]: ...
    def getobj(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Optional[Mapping[str, str]] = ...,
        fallback: object = ...,
        fallback_type: _T = ...,
        warn: bool = ...,
    ) -> Union[Any, _T]: ...
    def getlist(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Optional[Mapping[str, str]] = ...,
        fallback: _T = ...,
        element_type: _ET = ...,
    ) -> Union[_T, List[_ET]]: ...
    def getarray(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Optional[Mapping[str, str]] = ...,
        fallback: _T = ...,
    ) -> Union[np.ndarray, _T]: ...
    @classmethod
    def register_module(cls: _T, module): ...
    def jsonifyMetadata(self) -> str: ...
    def jsonifyAll(self) -> str: ...

class ConfigurableMixin(abc.ABC):
    @classmethod
    def get_config_options(cls, config: Config, name: str) -> Dict[str, Any]: ...
    @classmethod
    def from_config(cls, config: Config, name: Optional[str] = None): ...
