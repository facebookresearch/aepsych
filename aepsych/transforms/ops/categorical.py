#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any

import torch
from aepsych.config import Config
from aepsych.transforms.ops.base import StringParameterMixin, Transform


class Categorical(Transform, StringParameterMixin):
    # These attributes do nothing here but ensures compat.
    is_one_to_many = False
    transform_on_train = True
    transform_on_eval = True
    transform_on_fantasize = True
    training = True
    reverse = False

    def __init__(
        self,
        indices: list[int],
        categories: dict[int, list[str]],
    ) -> None:
        """Initialize a categorical transform. The transform itself does not
        change the tensors. Instead, this class allows passing in NumPy object
        arrays where the categorical values are stored as strings. This provides
        a convenient API to turn mixed categorical/continuous data into the
        expected form for models.

        Args:
            indices (list[int]): The indices of the inputs that are categorical.
            categories (dict[int, list[str]]): A dictionary mapping indices to
                the list of categories for that input. There must be a list for
                each index in `indices`.
        """
        self.indices = indices
        self.categories = categories
        self.string_map = self.categories

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        r"""This is a no-op as these transforms should be acting on indices
        already.

        Args:
            X (torch.Tensor): A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            torch.Tensor: The input tensor.
        """
        return X

    def _untransform(self, X: torch.Tensor) -> torch.Tensor:
        r"""This is a no-op as these transforms should be acting on indices
        already.

        Args:
            X (torch.Tensor): A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            torch.Tensor: The input tensor.
        """
        return X

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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

        if "categories" not in options:
            idx = options["indices"][0]  # There should only be one index
            cat_dict = {idx: config.getlist(name, "categories", element_type=str)}
            options["categories"] = cat_dict

        return options
