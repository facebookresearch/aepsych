#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import gpytorch
import torch
from aepsych.config import Config, ConfigurableMixin
from aepsych.factory.default import default_mean_covar_factory
from aepsych.utils import get_dims, get_optimizer_options, promote_0d
from aepsych.utils_logging import getLogger
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_scipy
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import TransformedPosterior
from gpytorch.mlls import MarginalLogLikelihood

logger = getLogger()


class AEPsychModelMixin(GPyTorchModel, ConfigurableMixin):
    """Mixin class that provides AEPsych-specific utility methods."""

    extremum_solver = "Nelder-Mead"
    outcome_types: List[str] = []
    _train_inputs: Optional[Tuple[torch.Tensor]]
    _train_targets: Optional[torch.Tensor]
    stimuli_per_trial: int = 1

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
            self._train_targets = train_targets.to(self.device)

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

    def predict(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            *args: Positional arguments for model-specific predict args.
            **kwargs: Keyword arguments for model-specific predict kwargs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at queries points.
        """
        with torch.no_grad():
            x = x.to(self.device)
            post = self.posterior(x)
            mean = post.mean.squeeze()
            var = post.variance.squeeze()

        return mean.to(self.device), var.to(self.device)

    def predict_transform(
        self,
        x: torch.Tensor,
        transformed_posterior_cls: Optional[type[TransformedPosterior]] = None,
        **transform_kwargs,
    ):
        """Query the model for posterior mean and variance under some tranformation.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            transformed_posterior_cls (TransformedPosterior type, optional): The type of transformation to apply to the posterior.
                Note that you should give TransformedPosterior itself, rather than an instance. Defaults to None, in which case no
                transformation is applied.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed posterior mean and variance at queries points.
        """
        if transformed_posterior_cls is None:
            return self.predict(x)
        with torch.no_grad():
            x = x.to(self.device)
            post = self.posterior(x)
            post = transformed_posterior_cls(post, **transform_kwargs)

            mean = post.mean.squeeze()
            var = post.variance.squeeze()

        return promote_0d(mean.to(self.device)), promote_0d(var.to(self.device))

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample from underlying model.

        Args:
            x (torch.Tensor): Points at which to sample.
            num_samples (int): Number of samples to return.
            kwargs are ignored

        Returns:
            torch.Tensor: Posterior samples [num_samples x dim]
        """
        x = x.to(self.device)
        return self.posterior(x).sample(torch.Size([num_samples])).squeeze()

    def update(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs):
        """Perform a warm-start update of the model from previous fit.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.Tensor): Responses.
        """
        return self.fit(train_x, train_y, **kwargs)

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): The name of the strategy to warm start (Not actually optional here.)
            options (Dict[str, Any], optional): options are ignored.

        Raises:
            ValueError: the name of the strategy is necessary to identify warm start search criteria.
            KeyError: the config specified this strategy should be warm started but the associated config section wasn't defined.

        Returns:
            Dict[str, Any]: a dictionary of the search criteria described in the experiment's config
        """
        name = name or cls.__name__
        options = super().get_config_options(config, name, options)

        dim = config.getint(name, "dim", fallback=None)
        if dim is None:
            dim = get_dims(config)

        mean_covar_factory = config.getobj(
            name, "mean_covar_factory", fallback=default_mean_covar_factory
        )

        mean, covar = mean_covar_factory(
            config, stimuli_per_trial=cls.stimuli_per_trial
        )
        max_fit_time = config.getfloat(name, "max_fit_time", fallback=None)

        likelihood_cls = config.getobj(name, "likelihood", fallback=None)

        if likelihood_cls is not None:
            if hasattr(likelihood_cls, "from_config"):
                likelihood = likelihood_cls.from_config(config)
            else:
                likelihood = likelihood_cls()
        else:
            likelihood = None  # fall back to __init__ default

        optimizer_options = get_optimizer_options(config, name)

        options.update(
            {
                "dim": dim,
                "mean_module": mean,
                "covar_module": covar,
                "max_fit_time": max_fit_time,
                "likelihood": likelihood,
                "optimizer_options": optimizer_options,
            }
        )

        return options
