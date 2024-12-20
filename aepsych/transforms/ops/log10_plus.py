#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from aepsych.config import Config
from aepsych.transforms.ops.base import Transform
from aepsych.utils import get_bounds
from botorch.models.transforms.input import Log10, subset_transform


class Log10Plus(Log10, Transform):
    """Base-10 log transform that we add a constant to the values"""

    def __init__(
        self,
        indices: List[int],
        constant: float = 0.0,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        **kwargs,
    ) -> None:
        """Initalize transform

        Args:
            indices (List[int]): The indices of the parameters to log transform.
            constant (float): The constant to add to inputs before log transforming.
                Defaults to 0.0.
            transform_on_train (bool): A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval (bool): A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize (bool): A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            reverse (bool): A boolean indicating whether the forward pass should
                untransform the inputs. Default: False.
            **kwargs: Accepted to conform to API.
        """
        super().__init__(
            indices=indices,
            transform_on_train=transform_on_train,
            transform_on_eval=transform_on_eval,
            transform_on_fantasize=transform_on_fantasize,
            reverse=reverse,
        )
        self.register_buffer("constant", torch.tensor(constant, dtype=torch.long))

    @subset_transform
    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        r"""Add the constant then log transform the inputs.

        Args:
            X (torch.Tensor): A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            torch.Tensor: A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        X = X + (torch.ones_like(X) * self.constant)
        return X.log10()

    @subset_transform
    def _untransform(self, X: torch.Tensor) -> torch.Tensor:
        r"""Reverse the log transformation then subtract the constant.

        Args:
            X (torch.Tensor): A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            torch.Tensor: A `batch_shape x n x d`-dim tensor of untransformed inputs.
        """
        X = 10.0**X
        return X - (torch.ones_like(X) * self.constant)

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
            name (str, optional): Parameter to find options for.
            options (Dict[str, Any], optional): Options to override from the config.

        Returns:
            Dict[str, Any]: A dictionary of options to initialize this class with,
                including the transformed bounds.
        """
        options = super().get_config_options(config=config, name=name, options=options)

        # Make sure we have bounds ready
        if "bounds" not in options:
            options["bounds"] = get_bounds(config)

        if "constant" not in options:
            lb = options["bounds"][0, options["indices"]]
            if lb < 0.0:
                constant = np.abs(lb) + 1.0
            elif lb < 1.0:
                constant = 1.0
            else:
                constant = 0.0

            options["constant"] = constant

        return options
