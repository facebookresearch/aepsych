#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Optional

import gpytorch
import torch
from aepsych.config import Config
from aepsych.models import GPRegressionModel


class MultitaskGPRModel(GPRegressionModel):
    """
    Multitask (multi-output) GP regression, using a kronecker-separable model
    where [a] each output is observed at each input, and [b] the kernel between
    two outputs at two points is given by k_x(x, x') * k_t[i, j] where k(x, x')
    is the usual GP kernel and k_t[i, j] is indexing into a freeform covariance
    of potentially low rank.

    This essentially implements / wraps the GPyTorch multitask GPR tutorial
    in https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html
    with AEPsych API and convenience fitting / prediction methods.
    """

    _num_outputs = 1
    _batch_size = 1
    stimuli_per_trial = 1
    outcome_type = "continuous"

    def __init__(
        self,
        num_outputs: int = 2,
        rank: int = 1,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize multitask GPR model.

        Args:
            num_outputs (int): Number of tasks (outputs). Defaults to 2.
            rank (int): Rank of cross-task covariance. Lower rank is a simpler model.
                Should be less than or equal to num_outputs. Defaults to 1.
            mean_module (gpytorch.means.Mean, optional): GP mean. Defaults to a constant mean.
            covar_module (gpytorch.kernels.Kernel, optional): GP kernel module.
                Defaults to scaled RBF kernel.
            likelihood (gpytorch.likelihoods.Likelihood, optional): Likelihood
                (should be a multitask-compatible likelihood). Defaults to multitask Gaussian likelihood.
        """
        self._num_outputs = num_outputs
        self.rank = rank

        likelihood = likelihood or gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self._num_outputs
        )
        super().__init__(
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            *args,
            **kwargs,
        )  # type: ignore # mypy issue 4335

        self.mean_module = gpytorch.means.MultitaskMean(
            self.mean_module, num_tasks=num_outputs
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            self.covar_module, num_tasks=num_outputs, rank=rank
        )

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        """Evaluate GP.

        Args:
            x (torch.Tensor): Tensor of points at which GP should be evaluated.

        Returns:
            gpytorch.distributions.MultitaskMultivariateNormal: Distribution object
                holding the mean and covariance at x."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    @classmethod
    def construct_inputs(cls, config: Config):
        """Construct inputs for the Multitask GPR model from configuration.

        Args:
            config (Config): A configuration containing keys/values matching this class.
        """
        classname = cls.__name__
        args = super().construct_inputs(config)
        args["num_outputs"] = config.getint(classname, "num_outputs", fallback=2)
        args["rank"] = config.getint(classname, "rank", fallback=1)
        return args


class IndependentMultitaskGPRModel(GPRegressionModel):
    """Independent multitask GP regression. This is a convenience wrapper for
    fitting a batch of independent GPRegression models. It wraps the GPyTorch tutorial here
    https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Batch_Independent_Multioutput_GP.html
    with AEPsych API and convenience fitting / prediction methods.
    """

    _num_outputs = 1
    _batch_size = 1
    stimuli_per_trial = 1
    outcome_type = "continuous"

    def __init__(
        self,
        num_outputs: int = 2,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize independent multitask GPR model.

        Args:
            num_outputs (int): Number of tasks (outputs). Defaults to 2.
            mean_module (gpytorch.means.Mean, optional): GP mean. Defaults to a constant mean.
            covar_module (gpytorch.kernels.Kernel, optional): GP kernel module.
                Defaults to scaled RBF kernel.
            likelihood (gpytorch.likelihoods.Likelihood, optional): Likelihood
                (should be a multitask-compatible likelihood). Defaults to multitask Gaussian likelihood.
        """

        self._num_outputs = num_outputs
        self._batch_size = num_outputs
        self._batch_shape = torch.Size([num_outputs])

        mean_module = mean_module or gpytorch.means.ConstantMean(
            batch_shape=self._batch_shape
        )

        covar_module = covar_module or gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=self._batch_shape),
            batch_shape=self._batch_shape,
        )

        likelihood = likelihood or gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self._batch_shape[0]
        )
        super().__init__(
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            *args,
            **kwargs,
        )  # type: ignore # mypy issue 4335

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        """Evaluate GP.

        Args:
            x (torch.Tensor): Tensor of points at which GP should be evaluated.

        Returns:
            gpytorch.distributions.MultitaskMultivariateNormal: Distribution object
                holding the mean and covariance at x.
        """
        base_mvn = super().forward(x)  # do transforms
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            base_mvn
        )

    @classmethod
    def get_config_args(cls, config: Config) -> Dict[str, Any]:
        """Get configuration arguments for the model.

        Args:
            config (Config): A configuration containing keys/values matching this class.

        Returns:
            Dict[str, Any]: Dictionary of configuration arguments.
        """
        classname = cls.__name__
        args = super().get_config_args(config)
        args["num_outputs"] = config.getint(classname, "num_outputs", fallback=2)
        return args
