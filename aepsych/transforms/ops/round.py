#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Literal, Optional

import torch
from aepsych.transforms.ops.base import Transform
from botorch.models.transforms.input import subset_transform


class Round(Transform, torch.nn.Module):
    def __init__(
        self,
        indices: List[int],
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a round transform. This operation rounds the inputs at the indices
        in both direction.

        Args:
            indices (List[int]): The indices of the inputs to round.
            transform_on_train (bool): A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval (bool): A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize (bool): Currently will not do anything, here to
                conform to API.
            reverse (bool): Whether to round in forward or backward passes. Does not do
                anything, both direction rounds.
            **kwargs: Accepted to conform to API.
        """
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.reverse = reverse

    @subset_transform
    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        r"""Round the inputs to a model to be discrete. This rounding is the same both
        in the forward and the backward pass.

        Args:
            X (torch.Tensor): A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            torch.Tensor: The input tensor with values rounded.
        """
        return X.round()

    @subset_transform
    def _untransform(self, X: torch.Tensor) -> torch.Tensor:
        r"""Round the inputs to a model to be discrete. This rounding is the same both
        in the forward and the backward pass.

        Args:
            X (torch.Tensor): A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            torch.Tensor: The input tensor with values rounded.
        """
        return X.round()

    def transform_bounds(
        self, X: torch.Tensor, bound: Optional[Literal["lb", "ub"]] = None, **kwargs
    ) -> torch.Tensor:
        r"""Return the bounds X transformed.

        Args:
            X (torch.Tensor): Either a `[1, dim]` or `[2, dim]` tensor of parameter
                bounds.
            bound (Literal["lb", "ub"], optional): The bound that this is, if None, we
                will assume the input is both bounds with a `[2, dim]` X.
            **kwargs: passed to _transform_bounds
                epsilon: will modify the offset for the rounding to ensure each discrete
                    value has equal space in the parameter space.

        Returns:
            torch.Tensor: A transformed set of parameter bounds.
        """
        epsilon = kwargs.get("epsilon", 1e-6)
        return self._transform_bounds(X, bound=bound, epsilon=epsilon)

    def _transform_bounds(
        self,
        X: torch.Tensor,
        bound: Optional[Literal["lb", "ub"]] = None,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        r"""Return the bounds X transformed.

        Args:
            X (torch.Tensor): Either a `[1, dim]` or `[2, dim]` tensor of parameter
                bounds.
            bound (Literal["lb", "ub"], optional): The bound that this is, if None, we
                will assume the input is both bounds with a `[2, dim]` X.
            epsilon:
            **kwargs: other kwargs

        Returns:
            torch.Tensor: A transformed set of parameter bounds.
        """
        X = X.clone()

        if bound == "lb":
            X[0, self.indices] -= torch.tensor([0.5] * len(self.indices))
        elif bound == "ub":
            X[0, self.indices] += torch.tensor([0.5 - epsilon] * len(self.indices))
        else:  # Both bounds
            X[0, self.indices] -= torch.tensor([0.5] * len(self.indices))
            X[1, self.indices] += torch.tensor([0.5 - epsilon] * len(self.indices))

        return X
