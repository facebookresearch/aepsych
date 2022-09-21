#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import Any, Dict, Optional, Union

import gpytorch
import numpy as np
import torch
from aepsych.config import Config
from aepsych.factory import default_mean_covar_factory
from aepsych.models.base import AEPsychMixin
from aepsych.utils import _process_bounds, promote_0d
from aepsych.utils_logging import getLogger
from botorch.fit import fit_gpytorch_mll
from botorch.models import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from scipy.stats import norm

logger = getLogger()


class PairwiseProbitModel(PairwiseGP, AEPsychMixin):
    _num_outputs = 1
    stimuli_per_trial = 2
    outcome_type = "binary"

    def _pairs_to_comparisons(self, x, y):
        """
        Takes x, y structured as pairs and judgments and
        returns pairs and comparisons as PairwiseGP requires
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
        return unique_coords.T, torch.LongTensor(comparisons)

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        dim: Optional[int] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        max_fit_time: Optional[float] = None,
    ):
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
            _, covar_module = default_mean_covar_factory(config)

        super().__init__(
            datapoints=None,
            comparisons=None,
            covar_module=covar_module,
            jitter=1e-3,
            input_transform=input_transform,
        )

        self.dim = dim  # The Pairwise constructor sets self.dim = None.

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.train()
        mll = PairwiseLaplaceMarginalLogLikelihood(self.likelihood, self)
        datapoints, comparisons = self._pairs_to_comparisons(train_x, train_y)
        self.set_train_data(datapoints, comparisons)

        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs.copy()
        max_fit_time = kwargs.pop("max_fit_time", self.max_fit_time)
        if max_fit_time is not None:
            # figure out how long evaluating a single samp
            starttime = time.time()
            _ = mll(self(datapoints), comparisons)
            single_eval_time = time.time() - starttime
            n_eval = int(max_fit_time / single_eval_time)
            optimizer_kwargs["maxfun"] = n_eval
            logger.info(f"fit maxfun is {n_eval}")

        logger.info("Starting fit...")
        starttime = time.time()
        fit_gpytorch_mll(mll, **kwargs, **optimizer_kwargs)
        logger.info(f"Fit done, time={time.time()-starttime}")

    def update(
        self, train_x: torch.Tensor, train_y: torch.Tensor, warmstart: bool = True
    ):
        """Perform a warm-start update of the model from previous fit."""
        self.fit(train_x, train_y)

    def predict(
        self, x, probability_space=False, num_samples=1000, rereference="x_min"
    ):
        if rereference is not None:
            samps = self.sample(x, num_samples, rereference)
            fmean, fvar = samps.mean(0).squeeze(), samps.var(0).squeeze()
        else:
            post = self.posterior(x)
            fmean, fvar = post.mean.squeeze(), post.variance.squeeze()

        if probability_space:
            return (
                promote_0d(norm.cdf(fmean)),
                promote_0d(norm.cdf(fvar)),
            )
        else:
            return fmean, fvar

    def sample(self, x, num_samples, rereference="x_min"):
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
    def from_config(cls, config):

        classname = cls.__name__

        mean_covar_factory = config.getobj(
            "PairwiseProbitModel",
            "mean_covar_factory",
            fallback=default_mean_covar_factory,
        )

        # no way of passing mean into PairwiseGP right now
        _, covar = mean_covar_factory(config)

        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = lb.shape[0]

        max_fit_time = config.getfloat(classname, "max_fit_time", fallback=None)

        return cls(lb=lb, ub=ub, dim=dim, covar_module=covar, max_fit_time=max_fit_time)
