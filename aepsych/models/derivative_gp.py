#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional, Union

import gpytorch
import torch
from aepsych.kernels.rbf_partial_grad import RBFKernelPartialObsGrad
from aepsych.means.constant_partial_grad import ConstantMeanPartialObsGrad
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.means import Mean
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class MixedDerivativeVariationalGP(gpytorch.models.ApproximateGP, GPyTorchModel):
    """A variational GP with mixed derivative observations.

    For more on GPs with derivative observations, see e.g. Riihimaki & Vehtari 2010.

    References:
        Riihimäki, J., & Vehtari, A. (2010). Gaussian processes with
            monotonicity information. Journal of Machine Learning Research, 9, 645–652.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        inducing_points: torch.Tensor,
        scales: Union[torch.Tensor, float] = 1.0,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        fixed_prior_mean: Optional[float] = None,
    ) -> None:
        """Initialize MixedDerivativeVariationalGP

        Args:
            train_x (torch.Tensor): Training x points. The last column of x is the derivative
                indiciator: 0 if it is an observation of f(x), and i if it
                is an observation of df/dx_i.
            train_y (torch.Tensor): Training y points
            inducing_points (torch.Tensor): Inducing points to use
            scales (Union[torch.Tensor, float]): Typical scale of each dimension
                of input space (this is used to set the lengthscale prior).
                Defaults to 1.0.
            mean_module (Mean, optional): A mean class that supports derivative
                indexes as the final dim. Defaults to a constant mean.
            covar_module (Kernel, optional): A covariance kernel class that
                supports derivative indexes as the final dim. Defaults to RBF kernel.
            fixed_prior_mean (float, optional): A prior mean value to use with the
                constant mean. Often setting this to the target threshold speeds
                up experiments. Defaults to None, in which case the mean will be inferred.
        """
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_distribution.to(train_x)
        variational_strategy = VariationalStrategy(
            model=self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=False,
        )
        super(MixedDerivativeVariationalGP, self).__init__(variational_strategy)

        # Set the mean if specified to
        if mean_module is None:
            self.mean_module = ConstantMeanPartialObsGrad()
        else:
            self.mean_module = mean_module

        if fixed_prior_mean is not None:
            self.mean_module.constant.requires_grad_(False)
            self.mean_module.constant.copy_(
                torch.tensor(fixed_prior_mean, dtype=train_x.dtype)
            )

        if covar_module is None:
            self.base_kernel = RBFKernelPartialObsGrad(
                ard_num_dims=train_x.shape[-1] - 1,
                lengthscale_prior=GammaPrior(3.0, 6.0 / scales),
            )
            self.covar_module = ScaleKernel(
                self.base_kernel, outputscale_prior=GammaPrior(2.0, 0.15)
            )
        else:
            self.covar_module = covar_module

        self._num_outputs = 1
        self.train_inputs = (train_x,)
        self.train_targets = train_y
        self(train_x)  # Necessary for CholeskyVariationalDistribution

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Evaluate the model

        Args:
            x (torch.Tensor): Points at which to evaluate.

        Returns:
            MultivariateNormal: Object containig mean and covariance
                of GP at these points.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
