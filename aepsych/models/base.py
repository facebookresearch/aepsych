#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple, Union

import gpytorch
import numpy as np
import torch
from aepsych.utils import dim_grid, get_jnd_multid, make_scaled_sobol
from aepsych.utils_logging import getLogger
from botorch.acquisition import PosteriorMean
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_scipy
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim import optimize_acqf
from botorch.posteriors import GPyTorchPosterior
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import MarginalLogLikelihood
from scipy.optimize import minimize
from scipy.stats import norm

logger = getLogger()

torch.set_default_dtype(torch.double)  # TODO: find a better way to prevent type errors


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

    def posterior(self, x: torch.Tensor) -> GPyTorchPosterior:
        pass

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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
    ) -> Tuple[float, np.ndarray]:
        pass

    def dim_grid(self, gridsize: int = 30) -> torch.Tensor:
        pass

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs: Any) -> None:
        pass

    def update(
        self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs: Any
    ) -> None:
        pass

    def p_below_threshold(self, x, f_thresh) -> np.ndarray:
        pass


class AEPsychMixin(GPyTorchModel):
    """Mixin class that provides AEPsych-specific utility methods."""

    extremum_solver = "Nelder-Mead"
    outcome_types: List[str] = []

    @property
    def bounds(self):
        return torch.stack((self.lb, self.ub))

    def _get_extremum(
        self: ModelProtocol,
        extremum_type: str,
        locked_dims: Optional[Mapping[int, List[float]]] = None,
        n_samples: int = 1000,
    ) -> Tuple[float, np.ndarray]:
        """Return the extremum (min or max) of the modeled function
        Args:
            extremum_type (str): type of extremum (currently 'min' or 'max'
            n_samples (int, optional): number of coarse grid points to sample for optimization estimate.
        Returns:
            Tuple[float, np.ndarray]: Tuple containing the min and its location (argmin).
        """
        locked_dims = locked_dims or {}

        acqf = PosteriorMean(model=self, maximize=(extremum_type == "max"))
        bounds = torch.stack((self.lb, self.ub))
        best_point, best_val = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=n_samples,
            fixed_features=locked_dims,
        )

        # PosteriorMean flips the sign on minimize, we flip it back
        if extremum_type == "min":
            best_val = -best_val
        return best_val, best_point.squeeze(0)

    def get_max(
        self: ModelProtocol,
        locked_dims: Optional[Mapping[int, List[float]]] = None,
    ) -> Tuple[float, np.ndarray]:
        """Return the maximum of the modeled function, subject to constraints
        Returns:
            Tuple[float, np.ndarray]: Tuple containing the max and its location (argmax).
            locked_dims (Mapping[int, List[float]]): Dimensions to fix, so that the
                inverse is along a slice of the full surface.
        """
        locked_dims = locked_dims or {}
        return self._get_extremum("max", locked_dims)

    def get_min(
        self: ModelProtocol,
        locked_dims: Optional[Mapping[int, List[float]]] = None,
    ) -> Tuple[float, np.ndarray]:
        """Return the minimum of the modeled function, subject to constraints
        Returns:
            Tuple[float, np.ndarray]: Tuple containing the min and its location (argmin).
            locked_dims (Mapping[int, List[float]]): Dimensions to fix, so that the
                inverse is along a slice of the full surface.
        """
        locked_dims = locked_dims or {}
        return self._get_extremum("min", locked_dims)

    def inv_query(
        self: ModelProtocol,
        y: float,
        locked_dims: Optional[Mapping[int, List[float]]] = None,
        probability_space: bool = False,
        n_samples: int = 1000,
    ) -> Tuple[float, torch.Tensor]:
        """Query the model inverse.
        Return nearest x such that f(x) = queried y, and also return the
            value of f at that point.
        Args:
            y (float): Points at which to find the inverse.
            locked_dims (Mapping[int, List[float]]): Dimensions to fix, so that the
                inverse is along a slice of the full surface.
            probability_space (bool, optional): Is y (and therefore the
                returned nearest_y) in probability space instead of latent
                function space? Defaults to False.
        Returns:
            Tuple[float, np.ndarray]: Tuple containing the value of f
                nearest to queried y and the x position of this value.
        """
        if probability_space:
            assert (
                self.outcome_type == "binary"
            ), f"Cannot get probability space for outcome_type '{self.outcome_type}'"

        locked_dims = locked_dims or {}

        def model_distance(x, pt, probability_space):
            return np.abs(
                self.predict(torch.tensor([x]), probability_space=probability_space)[0]
                .detach()
                .numpy()
                - pt
            )

        # Look for point with value closest to y, subject the dict of locked dims

        query_lb = self.lb.clone()
        query_ub = self.ub.clone()

        for locked_dim in locked_dims.keys():
            dim_values = locked_dims[locked_dim]
            if len(dim_values) == 1:
                query_lb[locked_dim] = dim_values[0]
                query_ub[locked_dim] = dim_values[0]
            else:
                query_lb[locked_dim] = dim_values[0]
                query_ub[locked_dim] = dim_values[1]

        d = make_scaled_sobol(query_lb, query_ub, n_samples, seed=0)

        bounds = zip(query_lb.numpy(), query_ub.numpy())

        fmean, _ = self.predict(d, probability_space=probability_space)

        f = torch.abs(fmean - y)
        estimate = d[torch.where(f == torch.min(f))[0][0]].numpy()
        a = minimize(
            model_distance,
            estimate,
            args=(y, probability_space),
            method=self.extremum_solver,
            bounds=bounds,
        )
        val = self.predict(torch.tensor([a.x]), probability_space=probability_space)[
            0
        ].item()
        return val, torch.Tensor(a.x)

    def get_jnd(
        self: ModelProtocol,
        grid: Optional[Union[np.ndarray, torch.Tensor]] = None,
        cred_level: Optional[float] = None,
        intensity_dim: int = -1,
        confsamps: int = 500,
        method: str = "step",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Calculate the JND.

        Note that JND can have multiple plausible definitions
        outside of the linear case, so we provide options for how to compute it.
        For method="step", we report how far one needs to go over in stimulus
        space to move 1 unit up in latent space (this is a lot of people's
        conventional understanding of the JND).
        For method="taylor", we report the local derivative, which also maps to a
        1st-order Taylor expansion of the latent function. This is a formal
        generalization of JND as defined in Weber's law.
        Both definitions are equivalent for linear psychometric functions.

        Args:
            grid (Optional[np.ndarray], optional): Mesh grid over which to find the JND.
                Defaults to a square grid of size as determined by aepsych.utils.dim_grid
            cred_level (float, optional): Credible level for computing an interval.
                Defaults to None, computing no interval.
            intensity_dim (int, optional): Dimension over which to compute the JND.
                Defaults to -1.
            confsamps (int, optional): Number of posterior samples to use for
                computing the credible interval. Defaults to 500.
            method (str, optional): "taylor" or "step" method (see docstring).
                Defaults to "step".

        Raises:
            RuntimeError: for passing an unknown method.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: either the
                mean JND, or a median, lower, upper tuple of the JND posterior.
        """
        if grid is None:
            grid = self.dim_grid()
        else:
            grid = torch.tensor(grid)

        # this is super awkward, back into intensity dim grid assuming a square grid
        gridsize = int(grid.shape[0] ** (1 / grid.shape[1]))
        coords = torch.linspace(
            self.lb[intensity_dim].item(), self.ub[intensity_dim].item(), gridsize
        )

        if cred_level is None:
            fmean, _ = self.predict(grid)
            fmean = fmean.reshape(*[gridsize for i in range(self.dim)])

            if method == "taylor":
                return torch.tensor(1 / np.gradient(fmean, coords, axis=intensity_dim))
            elif method == "step":
                return torch.clip(
                    torch.tensor(
                        get_jnd_multid(
                            fmean.detach().numpy(),
                            coords.detach().numpy(),
                            mono_dim=intensity_dim,
                        )
                    ),
                    0,
                    np.inf,
                )

        alpha = 1 - cred_level  # type: ignore
        qlower = alpha / 2
        qupper = 1 - alpha / 2

        fsamps = self.sample(grid, confsamps)
        if method == "taylor":
            jnds = torch.tensor(
                1
                / np.gradient(
                    fsamps.reshape(confsamps, *[gridsize for i in range(self.dim)]),
                    coords,
                    axis=intensity_dim,
                )
            )
        elif method == "step":
            samps = [s.reshape((gridsize,) * self.dim) for s in fsamps]
            jnds = torch.stack(
                [get_jnd_multid(s, coords, mono_dim=intensity_dim) for s in samps]
            )
        else:
            raise RuntimeError(f"Unknown method {method}!")
        upper = torch.clip(torch.quantile(jnds, qupper, axis=0), 0, np.inf)  # type: ignore
        lower = torch.clip(torch.quantile(jnds, qlower, axis=0), 0, np.inf)  # type: ignore
        median = torch.clip(torch.quantile(jnds, 0.5, axis=0), 0, np.inf)  # type: ignore
        return median, lower, upper

    def dim_grid(
        self: ModelProtocol,
        gridsize: int = 30,
        slice_dims: Optional[Mapping[int, float]] = None,
    ) -> torch.Tensor:
        return dim_grid(self.lb, self.ub, self.dim, gridsize, slice_dims)

    def set_train_data(self, inputs=None, targets=None, strict=False):
        """
        :param torch.Tensor inputs: The new training inputs.
        :param torch.Tensor targets: The new training targets.
        :param bool strict: (default False, ignored). Here for compatibility with
        input transformers. TODO: actually use this arg or change input transforms
        to not require it.
        """
        if inputs is not None:
            self.train_inputs = (inputs,)

        if targets is not None:
            self.train_targets = targets

    def normalize_inputs(self, x):
        scale = self.ub - self.lb
        return (x - self.lb) / scale

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Evaluate GP

        Args:
            x (torch.Tensor): Tensor of points at which GP should be evaluated.

        Returns:
            gpytorch.distributions.MultivariateNormal: Distribution object
                holding mean and covariance at x.
        """
        transformed_x = self.normalize_inputs(x)
        mean_x = self.mean_module(transformed_x)
        covar_x = self.covar_module(transformed_x)
        pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return pred

    def _fit_mll(
        self,
        mll: MarginalLogLikelihood,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        optimizer=fit_gpytorch_mll_scipy,
        **kwargs,
    ) -> None:
        self.train()
        train_x, train_y = mll.model.train_inputs[0], mll.model.train_targets
        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs.copy()
        max_fit_time = kwargs.pop("max_fit_time", self.max_fit_time)
        if max_fit_time is not None:
            # figure out how long evaluating a single samp
            starttime = time.time()
            _ = mll(self(train_x), train_y)
            single_eval_time = time.time() - starttime
            n_eval = int(max_fit_time / single_eval_time)
            optimizer_kwargs["options"] = {"maxfun": n_eval}
            logger.info(f"fit maxfun is {n_eval}")

        logger.info("Starting fit...")
        starttime = time.time()
        res = fit_gpytorch_mll(
            mll, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, **kwargs
        )
        logger.info(f"Fit done, time={time.time()-starttime}")
        return res

    def p_below_threshold(self, x, f_thresh) -> np.ndarray:
        f, var = self.predict(x)
        return norm.cdf((f_thresh - f.detach().numpy()) / var.sqrt().detach().numpy())
