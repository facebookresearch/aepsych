#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Union

import torch
from aepsych.config import Config
from aepsych.transforms.ops.base import StringParameterMixin, Transform


class Fixed(Transform, StringParameterMixin, torch.nn.Module):
    def __init__(
        self,
        indices: List[int],
        values: List[Union[float, int]],
        string_map: Optional[Dict[int, List[str]]] = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a fixed transform. It will add and remove fixed values from
        tensors.

        Args:
            indices (List[int]): The indices of the parameters to be fixed.
            values (List[Union[float, int]]): The values of the fixed parameters.
            string_map (Dict[int, List[str]], optional): A dictionary to allow some
                fixed elements to represent one element of a categorical parameter.
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
        # Turn indices and values into tensors and sort
        indices_ = torch.tensor(indices, dtype=torch.long)
        values_ = torch.tensor(values, dtype=torch.float64)

        # Sort indices and values
        sort_idx = torch.argsort(indices_)
        indices_ = indices_[sort_idx]
        values_ = values_[sort_idx]

        super().__init__()
        self.register_buffer("indices", indices_)
        self.register_buffer("values", values_)
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.reverse = reverse
        self.string_map = string_map

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        r"""Transform the input Tensor by popping out the fixed parameters at the
        specified indices.

        Args:
            X (torch.Tensor): A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            torch.Tensor: The input tensor with fixed parameters removed.
        """
        X = X.clone()

        mask = ~torch.isin(torch.arange(X.shape[1]), self.indices)

        X = X[:, mask]

        return X

    def _untransform(self, X: torch.Tensor) -> torch.Tensor:
        r"""Transform the input tensor by adding back in the fixed parameters at the
        specified indices.

        Args:
            X (torch.Tensor): A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            torch.Tensor: The same tensor as the input with the fixed parameters added
                back in.
        """
        X = X.clone()

        for i, idx in enumerate(self.indices):
            pre_fixed = X[:, :idx]
            post_fixed = X[:, idx:]
            fixed = torch.tile(self.values[i], (X.shape[0], 1))
            X = torch.cat((pre_fixed, fixed, post_fixed), dim=1)

        return X

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a dictionary of the relevant options to initialize a Fixed parameter
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

        if name is None:
            raise ValueError(f"{name} must be set to initialize a transform.")

        if "values" not in options:
            value = config[name].get("value")

            if value is None:
                raise ValueError(f"Value option not found in {name} section.")

            try:
                options["values"] = [float(value)]
            except ValueError:
                # Probably a string, so we treat it as categorical parameter fixed
                options["string_map"] = {options["indices"][0]: [value]}
                options["values"] = [0]

        return options
