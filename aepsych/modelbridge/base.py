
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from inspect import signature
from typing import Dict, Mapping, Optional, Tuple, Union

from aepsych.utils import make_scaled_sobol
import aepsych.utils_logging as utils_logging

import botorch

import gpytorch
import numpy as np
import torch
from aepsych.utils import _dim_grid, get_jnd_multid, promote_0d
from botorch.acquisition import (
    NoisyExpectedImprovement,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from scipy.optimize import minimize
from scipy.stats import norm

# this is pretty aggressive jitter setting but should protect us from
# crashes which are a bigger concern in data collection.
# we refit with stan afterwards anyway.
gpytorch.settings.cholesky_jitter._global_float_value = 1e-3
gpytorch.settings.cholesky_jitter._global_double_value = 1e-3
gpytorch.settings.tridiagonal_jitter._global_value = 1e-3

logger = utils_logging.getLogger(logging.INFO)


def _prune_extra_acqf_args(acqf, extra_acqf_args: Dict):
    # prune extra args needed, ignore the rest
    # (this helps with API consistency)
    acqf_args_expected = signature(acqf).parameters.keys()
    return {k: v for k, v in extra_acqf_args.items() if k in acqf_args_expected}


class ModelBridge(object):
    """Base class for objects combining an interpolator/model, acquisition, and data
    Loosely inspired by https://ax.dev/api/modelbridge.html#module-ax.modelbridge.base
    but definitely not compatible with it.

    Attributes:
        baseline_requiring_acqfs: list of acquisition functions that need an X_baseline
            passed in.
    """

    baseline_requiring_acqfs = [qNoisyExpectedImprovement, NoisyExpectedImprovement]
    model: gpytorch.models.GP

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        dim: int = 1,
        acqf: Optional[botorch.acquisition.AcquisitionFunction] = None,
        extra_acqf_args: Dict[str, object] = None,
    ):
        """Inititalize the base modelbridge class

        Args:
            lb (Union[np.ndarray, torch.Tensor]): Lower bounds of search space.
            ub (Union[np.ndarray, torch.Tensor]): Upper bounds of search space.
            dim (int, optional): Number of dimensions of search space. Defaults to 1.
            acqf (botorch.acquisition.AcquisitionFunction, optional): Acquisition function to
                use. Defaults to qUpperConfidenceBound.
            extra_acqf_args (Dict[str, object], optional): Additional arguments to pass to
                acquisition function. Defaults to nothing (except for qUCB, where we pass beta=1.96).
        """
        self.dim = dim

        self.lb = torch.Tensor(promote_0d(lb)).float()
        self.ub = torch.Tensor(promote_0d(ub)).float()
        if extra_acqf_args is None:
            extra_acqf_args = {}

        if acqf is None:
            self.acqf = qUpperConfidenceBound
            extra_acqf_args["beta"] = 1.96
        else:
            self.acqf = acqf

        self.extra_acqf_args = _prune_extra_acqf_args(self.acqf, extra_acqf_args)
        self.objective = extra_acqf_args.get("objective", None)
        self.target = extra_acqf_args.get("target", 0.75)

    def gen(self):
        """Generate next point to sample.

        Raises:
            NotImplementedError: Subclass from this class and implement this.
        """
        raise NotImplementedError("Implement me in subclasses!")

    def predict(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call underlying model for prediction.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Points at which to predict.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at x.
        Raises:
            NotImplementedError: Subclass from this class and implement this.
        """
        raise NotImplementedError("Implement me in subclasses!")

    def fit(self, train_x: torch.Tensor, train_y: torch.LongTensor):
        """Fit underlying model.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.LongTensor): Responses.
        Raises:
            NotImplementedError: Subclass from this class and implement this.
        """
        raise NotImplementedError("Implement me in subclasses!")

    def update(self, train_x: torch.Tensor, train_y: torch.LongTensor):
        """Perform a warm-start update of the model from previous fit.

        Note that in this base class, this just calls fit directly. Override
        this in subclasses to implement a more efficient model update.
        """
        logger.info(
            "Calling update on a model without specialized "
            + "update implementation, defaulting to regular fit"
        )
        self.fit(train_x, train_y)

    def sample(
        self, x: torch.Tensor, num_samples: int, **kwargs: object
    ) -> torch.Tensor:
        """Sample from the underlying model

        Args:
            x (torch.Tensor): Points at which to sample from model.
            num_samples (int): Number of samples to generate
            kwargs are ignored here but can be used in subclasses

        Returns:
            torch.Tensor: Samples from the model.
        """
        return self.model.posterior(x).sample(torch.Size([num_samples]))

    def _get_acquisition_fn(self):
        if self.acqf in self.baseline_requiring_acqfs:
            train_x = self.model.train_inputs[0]
            return self.acqf(
                model=self.model, X_baseline=train_x, **self.extra_acqf_args
            )
        else:
            return self.acqf(model=self.model, **self.extra_acqf_args)

    def best(self) -> Optional[np.ndarray]:
        """Return the current best.

        Note that currently this only returns the threshold for LSE
        acquisition functions, or otherwise nothing.

        TODO: make this actually return a more reasonable "best".

        Returns:
            np.ndarray: Current threshold estimate.
        """
        from aepsych.acquisition import lse_acqfs

        if self.acqf in lse_acqfs:
            return self._get_contour()
        else:
            return None

    def _get_contour(self, gridsize: int = 30) -> Optional[np.ndarray]:
        """Get a LSE contour from the underlying model.

        Currently only works in 2d, else returns None.

        Args:
            gridsize (int, optional): Number of grid points to evaluate threshold at. Defaults to 30.

        Returns:
            Optional[np.ndarray]: Threshold as a function of context.
        """

        from aepsych.utils import get_lse_contour

        if self.dim == 2:

            grid_search = _dim_grid(self, gridsize=gridsize)
            post_mean, _ = self.predict(torch.Tensor(grid_search))
            post_mean = norm.cdf(post_mean.reshape(gridsize, gridsize).detach().numpy())
            x1 = _dim_grid(lower=self.lb, upper=self.ub, dim=1, gridsize=gridsize)
            x2_hat = get_lse_contour(
                post_mean, x1, level=self.target, lb=x1.min(), ub=x1.max()
            )
            return x2_hat
        else:
            return None

    def get_jnd(
        self,
        grid: Optional[np.ndarray] = None,
        cred_level: float = None,
        intensity_dim: int = -1,
        confsamps: int = 500,
        method: str = "step",
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
                Defaults to a square grid of size as determined by aepsych.utils._dim_grid
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
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]: either the
                mean JND, or a median, lower, upper tuple of the JND posterior.
        """
        if grid is None:
            grid = _dim_grid(self)

        # this is super awkward, back into intensity dim grid assuming a square grid
        gridsize = int(np.power(grid.shape[0], 1 / grid.shape[1]))
        coords = np.linspace(self.lb[intensity_dim], self.ub[intensity_dim], gridsize)

        if cred_level is None:
            fmean, _ = self.predict(grid)
            fmean = fmean.detach().numpy().reshape(*[gridsize for i in range(self.dim)])

            if method == "taylor":
                return 1 / np.gradient(fmean, coords, axis=intensity_dim)
            elif method == "step":
                return np.clip(
                    get_jnd_multid(fmean, coords, mono_dim=intensity_dim), 0, np.inf
                )
        else:
            alpha = 1 - cred_level
            qlower = alpha / 2
            qupper = 1 - alpha / 2

            fsamps = self.sample(torch.Tensor(grid), confsamps)
            if method == "taylor":
                jnds = 1 / np.gradient(
                    fsamps.detach()
                    .numpy()
                    .reshape(confsamps, *[gridsize for i in range(self.dim)]),
                    coords,
                    axis=intensity_dim,
                )
            elif method == "step":
                samps = [
                    s.reshape((gridsize,) * self.dim) for s in fsamps.detach().numpy()
                ]
                jnds = np.stack(
                    [get_jnd_multid(s, coords, mono_dim=intensity_dim) for s in samps]
                )
            upper = np.clip(np.quantile(jnds, qupper, axis=0), 0, np.inf)
            lower = np.clip(np.quantile(jnds, qlower, axis=0), 0, np.inf)
            median = np.clip(np.quantile(jnds, 0.5, axis=0), 0, np.inf)
            return median, lower, upper

        raise RuntimeError(f"Unknown method {method}!")

    def query(
        self, x: torch.Tensor, probability_space: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            probability_space (bool, optional): Return outputs in units of
                response probability instead of latent function value. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Posterior mean and variance at queries points.
        """
        post = self.model.posterior(torch.Tensor(x))
        fmean, fvar = post.mean.squeeze().detach().numpy(), post.variance.squeeze().detach().numpy()
        if probability_space:
            return norm.cdf(fmean), norm.cdf(fvar)
        else:
            return fmean, fvar

    def get_max(self) -> Tuple[float, np.ndarray]:
        """Return the maximum of the modeled function

        Returns:
            Tuple[float, np.ndarray]: Tuple containing the max and its location (argmax).
        """
        return self._get_extremum('max')

    def _get_extremum(self, extremum_type: str, n_samples: int = 1000) -> Tuple[float, np.ndarray]:
        """Return the extremum (min or max) of the modeled function

        Args:
            extremum_type (str): type of extremum (currently 'min' or 'max'
            n_samples (int, optional): number of coarse grid points to sample for optimization estimate.

        Returns:
            Tuple[float, np.ndarray]: Tuple containing the min and its location (argmin).
        """

        def signed_model(x, sign=1):
            return sign*self.query([x])[0]

        bounds = tuple(zip(self.lb.numpy(), self.ub.numpy()))

        #generate a coarse sample to compute an initial estimate.
        d = make_scaled_sobol(self.lb, self.ub, n_samples, seed=0)

        fmean, _ = self.query(d)

        if extremum_type=='max':
            estimate = d[np.where(fmean==np.max(fmean))[0][0]]
            a = minimize(signed_model, estimate, args=-1, method='Powell', bounds=bounds)
            return -a.fun, a.x
        elif extremum_type=='min':
            estimate = d[np.where(fmean==np.min(fmean))[0][0]]
            a = minimize(signed_model, estimate, args=1, method='Powell', bounds=bounds)
            return a.fun, a.x

        else:
            raise RuntimeError(f"Unknown extremum type: '{extremum_type}'! Valid types: 'min', 'max' ")


    def get_min(self) -> Tuple[float, np.ndarray]:
        """Return the minimum of the modeled function

        Returns:
            Tuple[float, np.ndarray]: Tuple containing the min and its location (argmin).
        """
        return self._get_extremum('min')

    def inv_query(
        self,
        y: float,
        locked_dims: Mapping[int, float],
        probability_space: bool = False,
        n_samples: int = 1000
    ) -> Tuple[float, np.ndarray]:
        """Query the model inverse.

        Return nearest x such that f(x) = queried y, and also return the
            value of f at that point.

        Args:
            y (float): Points at which to find the inverse.
            locked_dims (Mapping[int, float]): Dimensions to fix, so that the
                inverse is along a slice of the full surface.
            probability_space (bool, optional): Is y (and therefore the
                returned nearest_y) in probability space instead of latent
                function space? Defaults to False.

        Returns:
            Tuple[float, np.ndarray]: Tuple containing the value of f
                nearest to queried y and the x position of this value.
        """

        def model_distance(x, pt, probability_space):
            return np.abs(self.query([x], probability_space)[0]-pt)

        #Look for point with value closest to y, subject the dict of locked dims

        query_lb = self.lb.clone().numpy()
        query_ub = self.ub.clone().numpy()

        for locked_dim in locked_dims.keys():
            query_lb[locked_dim] = locked_dims[locked_dim]
            query_ub[locked_dim] = locked_dims[locked_dim]

        d = make_scaled_sobol(query_lb, query_ub, n_samples, seed=0)

        bounds = tuple(zip(query_lb, query_ub))

        fmean, _ = self.query(d, probability_space)

        f = np.abs(fmean-y)
        estimate = d[np.where(f==np.min(f))[0][0]]
        a = minimize(model_distance, estimate, args=(y, probability_space), method='Powell', bounds=bounds)
        return self.query([a.x], probability_space)[0], a.x
