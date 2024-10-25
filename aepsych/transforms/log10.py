from botorch.models.transforms.input import Log10
from botorch.models.transforms.utils import subset_transform
import torch
from torch import Tensor


class Log10Plus(Log10):
    @subset_transform
    def _transform(self, X: Tensor) -> Tensor:
        r"""Add 1 then log transform the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        X = X + torch.ones_like(X)
        return X.log10()

    @subset_transform
    def _untransform(self, X: Tensor) -> Tensor:
        r"""Reverse the log transformation then subtract 1.

        Args:
            X: A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of untransformed inputs.
        """
        X = 10.0**X
        return X - torch.ones_like(X)
