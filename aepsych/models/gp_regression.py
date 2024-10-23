#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import gpytorch
import numpy as np
import torch
from aepsych.config import Config
from aepsych.factory.default import default_mean_covar_factory
from aepsych.models.base import AEPsychMixin
from aepsych.utils import _process_bounds, promote_0d
from aepsych.utils_logging import getLogger
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.models import ExactGP

logger = getLogger()


class GPRegressionModel(AEPsychMixin, ExactGP):
    """GP Regression model for continuous outcomes, using exact inference."""

    _num_outputs = 1
    _batch_size = 1
    stimuli_per_trial = 1
    outcome_type = "continuous"

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        dim: Optional[int] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        max_fit_time: Optional[float] = None,
        allow_gpu: bool = True,
    ):
        """Initialize the GP regression model

        Args:
            lb (Union[numpy.ndarray, torch.Tensor]): Lower bounds of the parameters.
            ub (Union[numpy.ndarray, torch.Tensor]): Upper bounds of the parameters.
            dim (int, optional): The number of dimensions in the parameter space. If None, it is inferred from the size
                of lb and ub.
            mean_module (gpytorch.means.Mean, optional): GP mean class. Defaults to a constant with a normal prior.
            covar_module (gpytorch.kernels.Kernel, optional): GP covariance kernel class. Defaults to scaled RBF with a
                gamma prior.
            likelihood (gpytorch.likelihood.Likelihood, optional): The likelihood function to use. If None defaults to
                Gaussian likelihood.
            max_fit_time (float, optional): The maximum amount of time, in seconds, to spend fitting the model. If None,
                there is no limit to the fitting time.
            allow_gpu (bool): If True, allows GPU to be used where speed-ups are likely. Defaults to True.
        """
        if allow_gpu:
            if not torch.cuda.is_available():
                self.allow_gpu = False
            else:
                self.allow_gpu = True
        else:
            self.allow_gpu = False

        # initial device of model is always cpu
        self.device = torch.device("cpu")

        if likelihood is None:
            likelihood = GaussianLikelihood()

        super().__init__(None, None, likelihood)

        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        self.max_fit_time = max_fit_time

        if mean_module is None or covar_module is None:
            default_mean, default_covar = default_mean_covar_factory(
                dim=self.dim, stimuli_per_trial=self.stimuli_per_trial
            )

        self.mean_module = mean_module or default_mean
        self.covar_module = covar_module or default_covar

        self._fresh_state_dict = deepcopy(self.state_dict())
        self._fresh_likelihood_dict = deepcopy(self.likelihood.state_dict())

    @classmethod
    def construct_inputs(cls, config: Config) -> Dict:
        classname = cls.__name__

        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = config.getint(classname, "dim", fallback=None)

        mean_covar_factory = config.getobj(
            classname, "mean_covar_factory", fallback=default_mean_covar_factory
        )

        mean, covar = mean_covar_factory(config)

        likelihood_cls = config.getobj(classname, "likelihood", fallback=None)

        if likelihood_cls is not None:
            if hasattr(likelihood_cls, "from_config"):
                likelihood = likelihood_cls.from_config(config)
            else:
                likelihood = likelihood_cls()
        else:
            likelihood = None  # fall back to __init__ default

        max_fit_time = config.getfloat(classname, "max_fit_time", fallback=None)

        allow_gpu = config.getboolean(classname, "allow_gpu", fallback=True)
        return {
            "lb": lb,
            "ub": ub,
            "dim": dim,
            "mean_module": mean,
            "covar_module": covar,
            "likelihood": likelihood,
            "max_fit_time": max_fit_time,
            "allow_gpu": allow_gpu,
        }

    @classmethod
    def from_config(cls, config: Config) -> GPRegressionModel:
        """Alternate constructor for GP regression model.

        This is used when we recursively build a full sampling strategy
        from a configuration. TODO: document how this works in some tutorial.

        Args:
            config (Config): A configuration containing keys/values matching this class

        Returns:
            GPRegressionModel: Configured class instance.
        """

        args = cls.construct_inputs(config)

        return cls(**args)

    def _move_device(self, device=None):
        # Moves important tensors (e.g., bounds/data) to a specific device. If device
        # is not set, it will automatically move tensors to the device most likely to
        # provide a speedup.
        if device is None:
            if self.allow_gpu:  # Initialized state overrides any automatic behavior
                device = "cuda"
            else:
                device = "cpu"

        # Change device for model when queried, otherwise doesn't do anything
        self.device = torch.device(device)

        self.lb = self.lb.to(torch.device(device))
        self.ub = self.ub.to(torch.device(device))
        self.likelihood = self.likelihood.to(torch.device(device))

        self.mean_module = self.mean_module.to(torch.device(device))
        self.covar_module = self.covar_module.to(torch.device(device))

        train_inputs = []
        for input in self.train_inputs:
            train_inputs.append(input.to(torch.device(device)))
        self.train_inputs = tuple(train_inputs)
        self.train_targets = self.train_targets.to(torch.device(device))

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs) -> None:
        """Fit underlying model.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.LongTensor): Responses.
        """
        self.set_train_data(train_x, train_y)
        self._move_device()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        return self._fit_mll(mll, **kwargs)

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample from underlying model.

        Args:
            x (torch.Tensor): Points at which to sample.
            num_samples (int, optional): Number of samples to return. Defaults to None.
            kwargs are ignored

        Returns:
            torch.Tensor: Posterior samples [num_samples x dim]
        """
        x = x.to(self.device)
        return self.posterior(x).rsample(torch.Size([num_samples])).detach().squeeze()

    def update(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs):
        """Perform a warm-start update of the model from previous fit."""
        return self.fit(train_x, train_y, **kwargs)

    def predict(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            probability_space (bool, optional): Return outputs in units of
                response probability instead of latent function value. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Posterior mean and variance at queries points.
        """
        with torch.no_grad():
            x = x.to(self.device)
            post = self.posterior(x)
        fmean = post.mean.squeeze()
        fvar = post.variance.squeeze()
        return promote_0d(fmean), promote_0d(fvar)
