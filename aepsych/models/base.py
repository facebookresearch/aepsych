#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple

import gpytorch
import torch
from aepsych.utils_logging import getLogger
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_scipy
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import GPyTorchPosterior
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import MarginalLogLikelihood

logger = getLogger()


class ModelProtocol(Protocol):
    @property
    def _num_outputs(self) -> int:
        pass

    @property
    def outcome_type(self) -> str:
        pass

    @property
    def extremum_solver(self) -> str:
        pass

    @property
    def train_inputs(self) -> torch.Tensor:
        pass

    @property
    def lb(self) -> torch.Tensor:
        pass

    @property
    def ub(self) -> torch.Tensor:
        pass

    @property
    def bounds(self) -> torch.Tensor:
        pass

    @property
    def dim(self) -> int:
        pass

    @property
    def device(self) -> torch.device:
        pass

    def posterior(self, X: torch.Tensor) -> GPyTorchPosterior:
        pass

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def predict_probability(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    @property
    def stimuli_per_trial(self) -> int:
        pass

    @property
    def likelihood(self) -> Likelihood:
        pass

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        pass

    def _get_extremum(
        self,
        extremum_type: str,
        locked_dims: Optional[Mapping[int, List[float]]],
        n_samples=1000,
    ) -> Tuple[float, torch.Tensor]:
        pass

    def dim_grid(self, gridsize: int = 30) -> torch.Tensor:
        pass

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs: Any) -> None:
        pass

    def update(
        self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs: Any
    ) -> None:
        pass

    def p_below_threshold(
        self, x: torch.Tensor, f_thresh: torch.Tensor
    ) -> torch.Tensor:
        pass


class AEPsychMixin(GPyTorchModel):
    """Mixin class that provides AEPsych-specific utility methods."""

    extremum_solver = "Nelder-Mead"
    outcome_types: List[str] = []
    train_inputs: Optional[Tuple[torch.Tensor]]
    train_targets: Optional[torch.Tensor]

    def set_train_data(
        self,
        inputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        strict: bool = False,
    ):
        """
        Set the training data for the model.

        Args:
            inputs (torch.Tensor, optional):  The new training inputs.
            targets (torch.Tensor, optional): The new training targets.
            strict (bool):  Default is False. Ignored, just for compatibility.

        input transformers. TODO: actually use this arg or change input transforms
        to not require it.
        """
        if inputs is not None:
            self.train_inputs = (inputs,)

        if targets is not None:
            self.train_targets = targets

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Evaluate GP

        Args:
            x (torch.Tensor): Tensor of points at which GP should be evaluated.

        Returns:
            gpytorch.distributions.MultivariateNormal: Distribution object
                holding mean and covariance at x.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return pred

    def _fit_mll(
        self,
        mll: MarginalLogLikelihood,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        optimizer: Callable = fit_gpytorch_mll_scipy,
        **kwargs,
    ) -> None:
        """Fits the model by maximizing the marginal log likelihood.

        Args:
            mll (MarginalLogLikelihood): Marginal log likelihood object.
            optimizer_kwargs (Dict[str, Any], optional): Keyword arguments for the optimizer.
            optimizer (Callable): Optimizer to use. Defaults to fit_gpytorch_mll_scipy.
        """
        self.train()
        train_x, train_y = mll.model.train_inputs[0], mll.model.train_targets
        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs.copy()
        max_fit_time = kwargs.pop("max_fit_time", self.max_fit_time)
        if max_fit_time is not None:
            if "options" not in optimizer_kwargs:
                optimizer_kwargs["options"] = {}

            # figure out how long evaluating a single samp
            starttime = time.time()
            _ = mll(self(train_x), train_y)
            single_eval_time = (
                time.time() - starttime + 1e-6
            )  # add an epsilon to avoid divide by zero
            n_eval = int(max_fit_time / single_eval_time)

            optimizer_kwargs["options"]["maxfun"] = n_eval
            logger.info(f"fit maxfun is {n_eval}")

        starttime = time.time()
        res = fit_gpytorch_mll(
            mll, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, **kwargs
        )
        return res

    def p_below_threshold(
        self, x: torch.Tensor, f_thresh: torch.Tensor
    ) -> torch.Tensor:
        """Compute the probability that the latent function is below a threshold.

        Args:
            x (torch.Tensor): Points at which to evaluate the probability.
            f_thresh (torch.Tensor): Threshold value.

        Returns:
            torch.Tensor: Probability that the latent function is below the threshold.
        """
        f, var = self.predict(x)
        f_thresh = f_thresh.reshape(-1, 1)
        f = f.reshape(1, -1)
        var = var.reshape(1, -1)

        z = (f_thresh - f) / var.sqrt()
        return torch.distributions.Normal(0, 1).cdf(z)  # Use PyTorch's CDF equivalent


class AEPsychModelDeviceMixin(AEPsychMixin):
    _train_inputs: Optional[Tuple[torch.Tensor]]
    _train_targets: Optional[torch.Tensor]

    def set_train_data(
        self,
        inputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        strict: bool = False,
    ) -> None:
        """Set the training data for the model.

        Args:
            inputs (torch.Tensor, optional): The new training inputs X.
            targets (torch.Tensor, optional): The new training targets Y.
            strict (bool): Whether to strictly enforce the device of the inputs and targets.

        input transformers. TODO: actually use this arg or change input transforms
        to not require it.
        """
        # Move to same device to ensure the right device
        if inputs is not None:
            self._train_inputs = (inputs.to(self.device),)

        if targets is not None:
            self._train_targets = targets.to(self.device)

    @property
    def device(self) -> torch.device:
        """Get the device of the model.

        Returns:
            torch.device: Device of the model.
        """
        # We assume all models have some parameters and all models will only use one device
        # notice that this has no setting, don't let users set device, use .to().
        return next(self.parameters()).device

    @property
    def train_inputs(self) -> Optional[Tuple[torch.Tensor]]:
        """Get the training inputs.

        Returns:
            Optional[Tuple[torch.Tensor]]: Training inputs.
        """
        if self._train_inputs is None:
            return None

        # makes sure the tensors are on the right device, move in place
        for input in self._train_inputs:
            input.to(self.device)

        return self._train_inputs

    @train_inputs.setter
    def train_inputs(self, train_inputs: Optional[Tuple[torch.Tensor]]) -> None:
        """Set the training inputs.

        Args:
            train_inputs (Tuple[torch.Tensor]): Training inputs.
        """
        if train_inputs is None:
            self._train_inputs = None
        else:
            # setting device on copy to not change original
            train_inputs = deepcopy(train_inputs)
            for input in train_inputs:
                input.to(self.device)

            self._train_inputs = train_inputs

    @property
    def train_targets(self) -> Optional[torch.Tensor]:
        """Get the training targets.

        Returns:
            Optional[torch.Tensor]: Training targets.
        """
        if self._train_targets is None:
            return None

        # make sure the tensors are on the right device
        self._train_targets = self._train_targets.to(self.device)

        return self._train_targets

    @train_targets.setter
    def train_targets(self, train_targets: Optional[torch.Tensor]) -> None:
        """Set the training targets.

        Args:
            train_targets (torch.Tensor, optional): Training targets.
        """
        if train_targets is None:
            self._train_targets = None
        else:
            # setting device on copy to not change original
            train_targets = deepcopy(train_targets).to(self.device)
            self._train_targets = train_targets
