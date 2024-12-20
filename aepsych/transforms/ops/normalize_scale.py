#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Union

import torch
from aepsych.config import Config
from aepsych.transforms.ops.base import Transform
from aepsych.utils import get_bounds
from botorch.models.transforms.input import Normalize


class NormalizeScale(Normalize, Transform):
    def __init__(
        self,
        d: int,
        indices: Optional[Union[List[int], torch.Tensor]] = None,
        bounds: Optional[torch.Tensor] = None,
        batch_shape: torch.Size = torch.Size(),
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        min_range: float = 1e-8,
        learn_bounds: Optional[bool] = None,
        almost_zero: float = 1e-12,
        **kwargs,
    ) -> None:
        r"""Normalizes the scale of the parameters.

        Args:
            d (int): Total number of parameters (dimensions).
            indices (Union[List[int], torch.Tensor], optional
                inputs to normalize. If omitted, take all dimensions of the inputs into
                account.
            bounds (torch.Tensor, optional): If provided, use these bounds to normalize
                the parameters. If omitted, learn the bounds in train mode.
            batch_shape (torch.Size): The batch shape of the inputs (assuming input
                tensors of shape `batch_shape x n x d`). If provided, perform individual
                normalization per batch, otherwise uses a single normalization.
            transform_on_train (bool): A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval (bool): A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize (bool): A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            reverse (bool): A boolean indicating whether the forward pass should
                untransform the parameters. Default: False.
            min_range (float): If the range of a parameter is smaller than `min_range`,
                that parameter will not be normalized. This is equivalent to
                using bounds of `[0, 1]` for this dimension, and helps avoid division
                by zero errors and related numerical issues. See the example below.
                NOTE: This only applies if `learn_bounds=True`. Defaults to 1e-8.
            learn_bounds (bool): Whether to learn the bounds in train mode. Defaults
                to False if bounds are provided, otherwise defaults to True.
            almost_zero (float): Threshold to consider the range essentially 0 and
                turns into a no op. Defaults to 1e-12.
            **kwargs: Accepted to conform to API.
        """
        super().__init__(
            d=d,
            indices=indices,
            bounds=bounds,
            batch_shape=batch_shape,
            transform_on_train=transform_on_train,
            transform_on_eval=transform_on_eval,
            transform_on_fantasize=transform_on_fantasize,
            reverse=reverse,
            min_range=min_range,
            learn_bounds=learn_bounds,
            almost_zero=almost_zero,
        )

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return a dictionary of the relevant options to initialize a NormalizeScale
        transform for the named parameter within the config.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): Parameter to find options for.
            options (Dict[str, Any], optional): Options to override from the config.

        Return:
            Dict[str, Any]: A dictionary of options to initialize this class with,
                including the transformed bounds.
        """
        options = super().get_config_options(config=config, name=name, options=options)

        # Make sure we have bounds ready
        if "bounds" not in options:
            options["bounds"] = get_bounds(config)

        if "d" not in options:
            options["d"] = options["bounds"].shape[1]

        return options
