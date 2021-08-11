#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple, Type, TypeVar, Union

import botorch
import gpytorch
import numpy as np
import torch
from aepsych.acquisition import MCLevelSetEstimation
from aepsych.acquisition.objective import ProbitObjective
from aepsych.config import Config
from aepsych.modelbridge.base import ModelBridge, _prune_extra_acqf_args
from aepsych.models import GPClassificationModel
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

ModelType = TypeVar("ModelType", bound="SingleProbitModelbridge")


class SingleProbitModelbridge(ModelBridge):
    """
    Modelbridge wrapping single probit GP models.

    Attributes:
        outcome_type: fixed to "single_probit".

    """

    outcome_type = "single_probit"

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        restarts: int = 10,
        samps: int = 1000,
        dim: int = 1,
        acqf: Optional[botorch.acquisition.AcquisitionFunction] = None,
        extra_acqf_args: Optional[Dict[str, object]] = None,
        model: Optional[gpytorch.models.GP] = None,
    ) -> None:
        """Initialize single probit modelbridge.

        Args:
            lb (Union[np.ndarray, torch.Tensor]): Lower bounds of domain.
            ub (Union[np.ndarray, torch.Tensor]): Upper bounds of domain.
            restarts (int, optional): Number of restarts for acquisition
                 function optimization. Defaults to 10.
            samps (int, optional): Number of quasi-random samples to use for finding
                an initial condition for acquisition function optimization. Defaults to 1000.
            dim (int, optional): Number of input dimensions. Defaults to 1.
            acqf (Optional[botorch.acquisition.AcquisitionFunction], optional): Acquisition
                function to use. Defaults to parent class default.
            extra_acqf_args (Dict[str, object], optional): Extra arguments to
                pass to acquisition function. Defaults to no arguments.
            model (gpytorch.models.GP, optional): Underlying model to use. Defaults
                to GPClassificationModel.
        """
        if extra_acqf_args is None:
            extra_acqf_args = {}

        super().__init__(
            lb=lb, ub=ub, dim=dim, acqf=acqf, extra_acqf_args=extra_acqf_args
        )

        self.restarts = restarts
        self.samps = samps

        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        if model is None:
            self.model = GPClassificationModel(
                inducing_min=self.lb, inducing_max=self.ub
            )
        else:
            self.model = model

    def fit(self, train_x: torch.Tensor, train_y: torch.LongTensor) -> None:
        """Fit underlying model.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.LongTensor): Responses.
        """
        n = train_y.shape[0]
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, n)
        self.model.train()
        self.model.set_train_data(train_x, train_y)
        fit_gpytorch_model(self.mll)

    def gen(self, num_points: int = 1, **kwargs):
        """Query next point(s) to run by optimizing the acquisition function.

        Args:
            num_points (int, optional): Number of points to query. Defaults to 1.
            Other arguments are forwared to underlying model.

        Returns:
            np.ndarray: Next set of point(s) to evaluate, [num_points x dim].
        """

        self.model.eval()
        train_x = self.model.train_inputs[0]
        acq = self._get_acquisition_fn()

        new_candidate, batch_acq_values = optimize_acqf(
            acq_function=acq,
            bounds=torch.tensor(np.c_[self.lb, self.ub]).T.to(train_x),
            q=num_points,
            num_restarts=self.restarts,
            raw_samples=self.samps,
        )

        return new_candidate.numpy()

    def predict(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call underlying model for prediction.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Points at which to predict.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at x.
        """
        post = self.model.posterior(torch.Tensor(x))
        return post.mean.squeeze(), post.variance.squeeze()

    def sample(self, x: torch.Tensor, num_samples: int, **kwargs) -> torch.Tensor:
        """Sample from underlying model.

        Args:
            x (torch.Tensor): Points at which to sample.
            num_samples (int, optional): Number of samples to return. Defaults to None.
            kwargs are ignored

        Returns:
            torch.Tensor: Posterior samples [num_samples x dim]
        """
        return self.model(x).rsample(torch.Size([num_samples]))

    @classmethod
    def from_config(cls: Type[ModelType], config: Config) -> ModelType:
        """Alternate constructor for this modelbridge from AEPsych config.

        Args:
            config (Config): Config containing params for this object and child objects.

        Returns:
            SingleProbitModelbridge: Configured class instance.
        """

        classname = cls.__name__
        model: GPClassificationModel = GPClassificationModel.from_config(config)

        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        restarts = config.getint(classname, "restarts", fallback=10)
        samps = config.getint(classname, "samps", fallback=1000)
        assert lb.shape[0] == ub.shape[0], "bounds are of different shapes!"
        dim = lb.shape[0]

        acqf = config.getobj("experiment", "acqf", fallback=MCLevelSetEstimation)
        acqf_name = acqf.__name__

        default_extra_acqf_args = {
            "beta": 3.98,
            "target": 0.75,
            "objective": ProbitObjective,
        }
        extra_acqf_args = {
            k: config.getobj(acqf_name, k, fallback_type=float, fallback=v, warn=False)
            for k, v in default_extra_acqf_args.items()
        }
        extra_acqf_args = _prune_extra_acqf_args(acqf, extra_acqf_args)
        if (
            "objective" in extra_acqf_args.keys()
            and extra_acqf_args["objective"] is not None
        ):
            extra_acqf_args["objective"] = extra_acqf_args["objective"]()
        return cls(
            lb=lb,
            ub=ub,
            restarts=restarts,
            samps=samps,
            dim=dim,
            acqf=acqf,
            model=model,
            extra_acqf_args=extra_acqf_args,
        )


class SingleProbitModelbridgeWithSongHeuristic(SingleProbitModelbridge):
    """
    Modelbridge that modifies the acquisition strategy with a heuristic
    given in Song et al. APP 2018 similar to Thompson sampling.

    Instead of optimizing the acquisition function, calling gen
    on this class instead evaluates the acquisition function on a
    quasi-random grid, normalizes it, adds noise, and picks the
    max of the noisy samples. This is not recommended in higher
    dimensions because the samples will be too sparse in the space,
    but improves exploration behavior in fewer dimensions.

    """

    def gen(
        self, num_points: int = 1, noise_scale: float = 0.2, **kwargs
    ) -> np.ndarray:
        """Query next point(s) to run using the heuristic from Song et al. 2018.

        Args:
            num_points (int, optional): Number of points to return. Defaults to 1.
            noise_scale (float, optional): Variance of the noise to add to the
                evaluated acquisition function before picking the max. Increasing
                this number will increase the amount of random exploration. Defaults to 0.2,
                the value from the paper.

        Returns:
            np.ndarray: Next set of point(s) to evaluate, [num_points x dim].
        """

        # Generate the points at which to sample
        X = draw_sobol_samples(
            bounds=torch.Tensor(np.c_[self.lb, self.ub]).T, n=self.samps, q=1
        ).squeeze(1)

        # Draw n samples
        f_samp = self.sample(X, num_samples=1000)
        acq = self._get_acquisition_fn()
        acq_vals = acq.acquisition(acq.objective(f_samp))
        # normalize
        acq_vals = acq_vals - acq_vals.min()
        acq_vals = acq_vals / acq_vals.max()
        # add noise
        acq_vals = acq_vals + torch.randn_like(acq_vals) * noise_scale

        # Find the point closest to target
        best_vals, best_indx = torch.topk(acq_vals, k=num_points)
        return X[best_indx]
