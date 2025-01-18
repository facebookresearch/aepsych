#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from aepsych.config import Config, ConfigurableMixin
from botorch.models.transforms.input import ReversibleInputTransform


class Transform(ReversibleInputTransform, ConfigurableMixin, ABC):
    """Base class for individual transforms. These transforms are intended to be stacked
    together using the ParameterTransforms class.
    """

    def transform_bounds(
        self, X: torch.Tensor, bound: Optional[Literal["lb", "ub"]] = None, **kwargs
    ) -> torch.Tensor:
        r"""Return the bounds X transformed.

        Args:
            X (torch.Tensor): Either a `[1, dim]` or `[2, dim]` tensor of parameter
                bounds.
            bound (Literal["lb", "ub"], optional): Which bound this is to transform, if
                None, it's the `[2, dim]` form with both bounds stacked.
            **kwargs: Keyword arguments for specific transforms, they should have
                default values.

        Returns:
            torch.Tensor: A transformed set of parameter bounds.
        """
        return self.transform(X)

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a dictionary of the relevant options to initialize a Log10Plus
        transform for the named parameter within the config.

        Args:
            config (Config): Config to look for options in.
            name (str, optionla): Parameter to find options for.
            options (Dict[str, Any], optional): Options to override from the config.

        Returns:
            Dict[str, Any]: A dictionary of options to initialize this class with,
                including the transformed bounds.
        """
        if name is None:
            raise ValueError(f"{name} must be set to initialize a transform.")

        if options is None:
            options = {}
        else:
            options = deepcopy(options)

        # Figure out the index of this parameter
        parnames = config.getlist("common", "parnames", element_type=str)
        idx = parnames.index(name)

        if "indices" not in options:
            options["indices"] = [idx]

        return options


class StringParameterMixin:
    string_map: Optional[Dict[int, List[str]]]

    def indices_to_str(self, X: np.ndarray) -> np.ndarray:
        r"""Return a NumPy array of objects where the parameter values that can be
        represented as a string is changed to a string.

        Args:
            X (np.ndarray): A mixed type NumPy array with some
                indices that will be turned into strings.

        Returns:
            np.ndarray: An array with the object type where the relevant parameters are
                converted to strings.
        """
        obj_arr = X.astype("O")

        if self.string_map is not None:
            for idx, cats in self.string_map.items():
                obj_arr[:, idx] = [cats[int(i)] for i in obj_arr[:, idx]]

        return obj_arr

    def str_to_indices(self, obj_arr: np.ndarray) -> np.ndarray:
        r"""Return a Tensor where the parameters represented by strings are converted
        into an index.

        Args:
            obj_arr (np.ndarray): A NumPy array `[batch, dim]` where the some parameters
                are strigns.

        Returns:
            np.ndarray: An array with the object type where the relevant string
                parameters are converted to indices
        """
        obj_arr = obj_arr[:]

        if self.string_map is not None:
            for idx, cats in self.string_map.items():
                obj_arr[:, idx] = [
                    cats.index(cat) if isinstance(cat, str) else cat
                    for cat in obj_arr[:, idx]
                ]

        return obj_arr
