#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import Any, Dict, Optional, Tuple

import gpytorch
import torch
from aepsych.config import Config
from aepsych.factory import default_mean_covar_factory
from aepsych.models.base import AEPsychModelMixin
from aepsych.utils import _process_bounds, get_dims, get_optimizer_options, promote_0d
from aepsych.utils_logging import getLogger
from botorch.fit import fit_gpytorch_mll
from botorch.models import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from torch.distributions import Normal

logger = getLogger()


class PairwiseProbitModel(PairwiseGP, AEPsychModelMixin):
    _num_outputs = 1
    stimuli_per_trial = 2
    outcome_type = "binary"

    def _pairs_to_comparisons(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert pairs of points and their judgements to comparisons.

        Args:
            x (torch.Tensor): Tensor of shape (n, d, 2) where n is the number of pairs and d is the dimensionality of the
                parameter space.
            y (torch.Tensor): Tensor of shape (n,) where n is the number of pairs. Each element is 0 if the first point
                in the pair is preferred, and 1 if the second point is preferred.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors. The first tensor is of shape (n, d) and contains the
                unique points in the pairs. The second tensor is of shape (n, 2) and contains the indices of the unique
                points in the first tensor that correspond to the points in the pairs.
        """
        # This needs to take a unique over the feature dim by flattening
        # over pairs but not instances/batches. This is actually tensor
        # matricization over the feature dimension but awkward in numpy
        unique_coords = torch.unique(
            torch.transpose(x, 1, 0).reshape(self.dim, -1), dim=1
        )

        def _get_index_of_equal_row(arr, x, dim=0):
            return torch.all(torch.eq(arr, x[:, None]), dim=dim).nonzero().item()

        comparisons = []
        for pair, judgement in zip(x, y):
            comparison = (
                _get_index_of_equal_row(unique_coords, pair[..., 0]),
                _get_index_of_equal_row(unique_coords, pair[..., 1]),
            )
            if judgement == 0:
                comparisons.append(comparison)
            else:
                comparisons.append(comparison[::-1])

        datapoints = unique_coords.T.to(self.device)
        comps = torch.LongTensor(comparisons).to(self.device)
        return datapoints, comps

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        dim: Optional[int] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        max_fit_time: Optional[float] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the PairwiseProbitModel

        Args:
            lb (torch.Tensor): Lower bounds of the parameters.
            ub (torch.Tensor): Upper bounds of the parameters.
            dim (int, optional): The number of dimensions in the parameter space. If None, it is inferred from the size
                of lb and ub. Defaults to None.
            covar_module (gpytorch.kernels.Kernel, optional): GP covariance kernel class. Defaults to scaled RBF with a
                gamma prior. Defaults to None.
            max_fit_time (float, optional): The maximum amount of time, in seconds, to spend fitting the model. Defaults to None.
        """
        self.lb, self.ub, dim = _process_bounds(lb, ub, dim)

        self.max_fit_time = max_fit_time

        bounds = torch.stack((self.lb, self.ub))
        input_transform = Normalize(d=dim, bounds=bounds)
        if covar_module is None:
            config = Config(
                config_dict={
                    "default_mean_covar_factory": {
                        "lb": str(self.lb.tolist()),
                        "ub": str(self.ub.tolist()),
                    }
                }
            )  # type: ignore
            _, covar_module = default_mean_covar_factory(
                config, stimuli_per_trial=self.stimuli_per_trial
            )

        super().__init__(
            datapoints=None,
            comparisons=None,
            covar_module=covar_module,
            jitter=1e-3,
            input_transform=input_transform,
        )

        self.dim = dim  # The Pairwise constructor sets self.dim = None.
        self.optimizer_options = (
            {"options": optimizer_options} if optimizer_options else {"options": {}}
        )

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Fit the model to the training data.

        Args:
            train_x (torch.Tensor): Trainin x points.
            train_y (torch.Tensor): Training y points.
            optimizer_kwargs (Dict[str, Any], optional): Keyword arguments to pass to the optimizer. Defaults to None.
        """
        if optimizer_kwargs is not None:
            if not "optimizer_kwargs" in optimizer_kwargs:
                optimizer_kwargs = optimizer_kwargs.copy()
                optimizer_kwargs.update(self.optimizer_options)
        else:
            optimizer_kwargs = {"options": self.optimizer_options}

        self.train()
        mll = PairwiseLaplaceMarginalLogLikelihood(self.likelihood, self)
        datapoints, comparisons = self._pairs_to_comparisons(train_x, train_y)
        self.set_train_data(datapoints, comparisons)

        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs.copy()
        max_fit_time = kwargs.pop("max_fit_time", self.max_fit_time)
        if max_fit_time is not None:
            if "options" not in optimizer_kwargs:
                optimizer_kwargs["options"] = {}

            # figure out how long evaluating a single samp
            starttime = time.time()
            _ = mll(self(datapoints), comparisons)
            single_eval_time = (
                time.time() - starttime + 1e-6
            )  # add an epsilon to avoid divide by zero
            n_eval = int(max_fit_time / (single_eval_time))

            optimizer_kwargs["options"]["maxfun"] = n_eval
            logger.info(f"fit maxfun is {n_eval}")

        logger.info("Starting fit...")
        starttime = time.time()
        fit_gpytorch_mll(mll, optimizer_kwargs=optimizer_kwargs, **kwargs)
        logger.info(f"Fit done, time={time.time() - starttime}")

    def predict(
        self,
        x: torch.Tensor,
        probability_space: bool = False,
        num_samples: int = 1000,
        rereference: str = "x_min",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            probability_space (bool): Return outputs in units of response probability instead of latent function value. Defaults to False.
            num_samples (int): Number of samples to return. Defaults to 1000.
            rereference (str): How to sample. Options are "x_min", "x_max", "f_min", "f_max". Defaults to "x_min".

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at query points.
        """
        if rereference is not None:
            samps = self.sample(x, num_samples, rereference)
            fmean, fvar = samps.mean(0).squeeze(), samps.var(0).squeeze()
        else:
            post = self.posterior(x)
            fmean, fvar = post.mean.squeeze(), post.variance.squeeze()

        if probability_space:
            return (
                promote_0d(Normal(0, 1).cdf(fmean.detach().cpu())),
                promote_0d(Normal(0, 1).cdf(fvar.detach().cpu())),
            )
        else:
            return fmean, fvar

    def predict_probability(
        self,
        x: torch.Tensor,
        probability_space: bool = False,
        num_samples: int = 1000,
        rereference: str = "x_min",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance in probability space.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            probability_space (bool): Return outputs in units of response probability instead of latent function value. Defaults to False.
            num_samples (int): Number of samples to return. Defaults to 1000.
            rereference (str): How to sample. Options are "x_min", "x_max", "f_min", "f_max". Defaults to "x_min".

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at query points.
        """
        return self.predict(
            x, probability_space=True, num_samples=num_samples, rereference=rereference
        )

    def sample(
        self, x: torch.Tensor, num_samples: int, rereference: str = "x_min"
    ) -> torch.Tensor:
        """Sample from the model model posterior.

        Args:
            x (torch.Tensor): Points at which to sample.
            num_samples (int): Number of samples to return.
            rereference (str): How to sample. Options are "x_min", "x_max", "f_min", "f_max". Defaults to "x_min".

        Returns:
            torch.Tensor: Posterior samples [num_samples x dim]
        """
        if len(x.shape) < 2:
            x = x.reshape(-1, 1)
        if rereference is None:
            return self.posterior(x).rsample(torch.Size([num_samples]))

        if rereference == "x_min":
            x_ref = self.lb
        elif rereference == "x_max":
            x_ref = self.ub
        elif rereference == "f_max":
            x_ref = torch.Tensor(self.get_max()[1])
        elif rereference == "f_min":
            x_ref = torch.Tensor(self.get_min()[1])
        else:
            raise RuntimeError(
                f"Unknown rereference type {rereference}! Options: x_min, x_max, f_min, f_max."
            )

        x_stack = torch.vstack([x, x_ref])
        samps = self.posterior(x_stack).rsample(torch.Size([num_samples]))
        samps, samps_ref = torch.split(samps, [samps.shape[1] - 1, 1], dim=1)
        if rereference == "x_min" or rereference == "f_min":
            return samps - samps_ref
        else:
            return -samps + samps_ref

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Alternate constructor for GPClassification model from a configuration.

        This is used when we recursively build a full sampling strategy
        from a configuration. TODO: document how this works in some tutorial.

        Args:
            config (Config): A configuration containing keys/values matching this class

        Returns:
            GPClassificationModel: Configured class instance.
        """
        options = super().get_config_options(config, name, options)

        # no way of passing mean into PairwiseGP right now
        if "mean_module" in options:
            del options["mean_module"]

        # This model doesn't take flexible likelihoods
        if "likelihood" in options:
            del options["likelihood"]

        return options
