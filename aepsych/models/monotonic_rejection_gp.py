#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

import torch
from aepsych.acquisition.monotonic_rejection import (
    MonotonicMCAcquisition,
    MonotonicMCLSE,
)
from aepsych.acquisition.objective import ProbitObjective
from aepsych.acquisition.rejection_sampler import RejectionSampler
from aepsych.models.derivative_gp import MixedDerivativeVariationalGP
from botorch.acquisition.monte_carlo import MCAcquisitionObjective
from botorch.acquisition.objective import IdentityMCObjective
from botorch.fit import fit_gpytorch_model
from botorch.logging import logger
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim.utils import columnwise_clamp, fix_features
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood
from gpytorch.means import Mean
from gpytorch.mlls.variational_elbo import VariationalELBO
from scipy.stats import norm
from torch import Tensor


def default_loss_constraint_fun(
    loss: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """Identity transform for constrained optimization.

    This simply returns loss as-is. Write your own versions of this
    for constrained optimization by e.g. interior point method.

    Args:
        loss (torch.Tensor): Value of loss at candidate points.
        candidates (torch.Tensor): Location of candidate points.

    Returns:
        torch.Tensor: New loss (unchanged)
    """
    return loss


class MonotonicRejectionGP:
    """A monotonic GP using rejection sampling.

    This takes the same insight as in e.g. Riihimäki & Vehtari 2010 (that the derivative of a GP
    is likewise a GP) but instead of approximately optimizing the likelihood of the model
    using EP, we optimize an unconstrained model by VI and then draw monotonic samples
    by rejection sampling.

    References:
        Riihimäki, J., & Vehtari, A. (2010). Gaussian processes with monotonicity information.
            Journal of Machine Learning Research, 9, 645–652.
    """

    def __init__(
        self,
        likelihood: str,
        monotonic_idxs: Sequence[int],
        fixed_prior_mean: Optional[float] = None,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        num_induc: int = 25,
        num_samples: int = 250,
        num_rejection_samples: int = 5000,
        acqf: MonotonicMCAcquisition = MonotonicMCLSE,
        objective: Optional[Union[MCAcquisitionObjective, object]] = None,
        extra_acqf_args: Optional[Dict[str, object]] = None,
    ) -> None:
        """Initialize MonotonicRejectionGP.

        Args:
            likelihood (str): Link function and likelihood. Can be 'probit-bernoulli' or
                'identity-gaussian'.
            monotonic_idxs (List[int]): List of which columns of x should be given monotonicity
            constraints.
            fixed_prior_mean (Optional[float], optional): Fixed prior mean. If classification, should be the prior
            classification probability (not the latent function value). Defaults to None.
            covar_module (Optional[Kernel], optional): Covariance kernel to use (default: scaled RBF).
            mean_module (Optional[Mean], optional): Mean module to use (default: constant mean).
            num_induc (int, optional): Number of inducing points for variational GP.]. Defaults to 25.
            num_samples (int, optional): Number of samples for estimating posterior on preDict or
            acquisition function evaluation. Defaults to 250.
            num_rejection_samples (int, optional): Number of samples used for rejection sampling. Defaults to 4096.
            acqf (MonotonicMCAcquisition, optional): Acquisition function to use for querying points. Defaults to MonotonicMCLSE.
            objective (Optional[MCAcquisitionObjective], optional): Transformation of GP to apply before computing acquisition function. Defaults to identity transform for gaussian likelihood, probit transform for probit-bernoulli.
            extra_acqf_args (Optional[Dict[str, object]], optional): Additional arguments to pass into the acquisition function. Defaults to None.
        """
        assert likelihood in ["probit-bernoulli", "identity-gaussian"]
        self.likelihood = likelihood
        self.num_induc = num_induc
        self.monotonic_idxs = monotonic_idxs
        self.num_samples = num_samples
        self.num_rejection_samples = num_rejection_samples
        self.fixed_prior_mean = fixed_prior_mean
        # These attributes are set during fitting
        self.device = None
        self.dtype = None
        self.inducing_points = None
        self.model = None
        self.model_likelihood = None
        self.bounds_ = None
        self.acqf = acqf

        if objective is None and likelihood == "probit-bernoulli":
            self.objective = ProbitObjective()
        elif objective is None and likelihood == "identity-gaussian":
            self.objective = IdentityMCObjective()
        else:
            self.objective = objective

        if extra_acqf_args is None:
            self.extra_acqf_args = {}
        else:
            self.extra_acqf_args = extra_acqf_args

        self.covar_module = covar_module
        self.mean_module = mean_module

    def fit(
        self, train_x: Tensor, train_y: Tensor, bounds: List[Tuple[float, float]]
    ) -> None:
        """Fit the model

        Args:
            train_x (Tensor): Training x points
            train_y (Tensor): Training y points. Should be (n x 1).
            bounds (List[Tuple[float, float]]): List of (lb, ub) tuples for each column in X.
        """
        self.dtype = train_x.dtype
        self.device = train_x.device
        bounds_ = torch.tensor(bounds, dtype=self.dtype)
        self.bounds_ = bounds_.transpose(0, 1)
        # Select inducing points
        self.inducing_points = draw_sobol_samples(
            bounds=self.bounds_, n=self.num_induc, q=1
        ).squeeze(1)
        self._set_model(train_x, train_y)

    def _set_model(
        self,
        train_x: Tensor,
        train_y: Tensor,
        model_state_dict: Optional[Dict[str, Tensor]] = None,
        likelihood_state_dict: Optional[Dict[str, Tensor]] = None,
    ) -> None:
        # Augment the data with the derivative index
        train_x_aug = self._augment_with_deriv_index(train_x, 0)
        inducing_points_aug = self._augment_with_deriv_index(self.inducing_points, 0)
        # Create and fit the model
        scales = self.bounds_[1, :] - self.bounds_[0, :]
        fixed_prior_mean = self.fixed_prior_mean
        if fixed_prior_mean is not None and self.likelihood == "probit-bernoulli":
            fixed_prior_mean = norm.ppf(fixed_prior_mean)
        self.model = MixedDerivativeVariationalGP(
            train_x=train_x_aug,
            train_y=train_y.squeeze(),
            inducing_points=inducing_points_aug,
            scales=scales,
            fixed_prior_mean=fixed_prior_mean,
            covar_module=self.covar_module,
            mean_module=self.mean_module,
        )

        self.model_likelihood = (
            BernoulliLikelihood()
            if self.likelihood == "probit-bernoulli"
            else GaussianLikelihood()
        )
        # Set model parameters
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
        if likelihood_state_dict is not None:
            self.model_likelihood.load_state_dict(likelihood_state_dict)

        # Fit!
        mll = VariationalELBO(
            likelihood=self.model_likelihood, model=self.model, num_data=train_y.numel()
        )
        mll = fit_gpytorch_model(mll)

    def update(self, train_x: Tensor, train_y: Tensor, warmstart: bool = True) -> None:
        """
        Update the model with new data.

        Expects the full set of data, not the incremental new data.

        Args:
            train_x (Tensor): Train X.
            train_y (Tensor): Train Y. Should be (n x 1).
            warmstart (bool): If True, warm-start model fitting with current parameters.
        """
        if warmstart:
            model_state_dict = self.model.state_dict()
            likelihood_state_dict = self.model_likelihood.state_dict()
        else:
            model_state_dict = None
            likelihood_state_dict = None
        self._set_model(
            train_x=train_x,
            train_y=train_y,
            model_state_dict=model_state_dict,
            likelihood_state_dict=likelihood_state_dict,
        )

    def sample(
        self,
        X: Tensor,
        num_samples: Optional[int] = None,
        num_rejection_samples: Optional[int] = None,
    ) -> Tensor:
        """Sample from monotonic GP

        Args:
            X (Tensor): tensor of n points at which to sample
            num_samples (int, optional): how many points to sample (default: self.num_samples)

        Returns: a Tensor of shape [n_samp, n]
        """
        if num_samples is None:
            num_samples = self.num_samples
        if num_rejection_samples is None:
            num_rejection_samples = self.num_rejection_samples

        rejection_ratio = 20
        if num_samples * rejection_ratio > num_rejection_samples:
            warnings.warn(
                f"num_rejection_samples should be at least {rejection_ratio} times greater than num_samples."
            )

        n = X.shape[0]
        # Augment with derivative index
        x_aug = self._augment_with_deriv_index(X, 0)
        # Add in monotonicity constraint points
        deriv_cp = self._get_deriv_constraint_points()
        x_aug = torch.cat((x_aug, deriv_cp), dim=0)
        assert x_aug.shape[0] == X.shape[0] + len(self.monotonic_idxs * self.num_induc)
        constrained_idx = torch.arange(n, x_aug.shape[0])

        with torch.no_grad():
            posterior = self.model.posterior(x_aug)
        sampler = RejectionSampler(
            num_samples=num_samples,
            num_rejection_samples=num_rejection_samples,
            constrained_idx=constrained_idx,
        )
        samples = sampler(posterior)
        samples_f = samples[:, :n, 0].detach().cpu()
        return samples_f

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict

        Args:
            X: tensor of n points at which to predict.

        Returns: tuple (f, var) where f is (n,) and var is (n,)
        """
        samples_f = self.sample(X)
        mean = torch.mean(samples_f, dim=0).squeeze()
        variance = torch.var(samples_f, dim=0).clamp_min(0).squeeze()
        return mean, variance

    def gen(
        self,
        model_gen_options: Optional[Dict[str, Any]] = None,
        explore_features: Optional[Sequence[int]] = None,
    ) -> Tuple[Tensor, Optional[List[Dict[str, Any]]]]:
        """Generate candidate by optimizing acquisition function.

        Args:
            model_gen_options: Dictionary with options for generating candidate, such as
                SGD parameters. See code for all options and their defaults.
            explore_features: List of features that will be selected randomly and then
                fixed for acquisition fn optimization.

        Returns:
            Xopt: (1 x d) tensor of the generated candidate
            candidate_metadata: List of dict of metadata for each candidate. Contains
                acquisition value for the candidate.
        """
        # Default optimization settings
        # TODO are these sufficiently robust? Can they be tuned better?
        options = model_gen_options or {}
        num_restarts = options.get("num_restarts", 10)
        raw_samples = options.get("raw_samples", 1000)
        verbosity_freq = options.get("verbosity_freq", -1)
        lr = options.get("lr", 0.01)
        momentum = options.get("momentum", 0.9)
        nesterov = options.get("nesterov", True)
        epochs = options.get("epochs", 50)
        milestones = options.get("milestones", [25, 40])
        gamma = options.get("gamma", 0.1)
        loss_constraint_fun = options.get(
            "loss_constraint_fun", default_loss_constraint_fun
        )

        acq_function = self._get_acquisition_fn()
        # Augment bounds with deriv indicator
        bounds = torch.cat((self.bounds_, torch.zeros(2, 1, dtype=self.dtype)), dim=1)
        # Fix deriv indicator to 0 during optimization
        fixed_features = {(bounds.shape[1] - 1): 0.0}
        # Fix explore features to random values
        if explore_features is not None:
            for idx in explore_features:
                val = (
                    bounds[0, idx]
                    + torch.rand(1, dtype=self.dtype)
                    * (bounds[1, idx] - bounds[0, idx])
                ).item()
                fixed_features[idx] = val
                bounds[0, idx] = val
                bounds[1, idx] = val

        # Initialize
        batch_initial_conditions = gen_batch_initial_conditions(
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        clamped_candidates = columnwise_clamp(
            X=batch_initial_conditions, lower=bounds[0], upper=bounds[1]
        ).requires_grad_(True)
        candidates = fix_features(clamped_candidates, fixed_features)
        optimizer = torch.optim.SGD(
            params=[clamped_candidates], lr=lr, momentum=momentum, nesterov=nesterov
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )

        # Optimize
        for epoch in range(epochs):
            loss = -acq_function(candidates).sum()

            # adjust loss based on constraints on candidates
            loss = loss_constraint_fun(loss, candidates)

            if verbosity_freq > 0 and epoch % verbosity_freq == 0:
                logger.info("Iter: {} - Value: {:.3f}".format(epoch, -(loss.item())))

            def closure():
                optimizer.zero_grad()
                loss.backward(
                    retain_graph=True
                )  # Variational model requires retain_graph
                return loss

            optimizer.step(closure)
            clamped_candidates.data = columnwise_clamp(
                X=clamped_candidates, lower=bounds[0], upper=bounds[1]
            )
            candidates = fix_features(clamped_candidates, fixed_features)
            lr_scheduler.step()

        # Extract best point
        with torch.no_grad():
            batch_acquisition = acq_function(candidates)
        best = torch.argmax(batch_acquisition.view(-1), dim=0)
        Xopt = candidates[best][:, :-1].detach()
        candidate_metadata = [{"acquisition_value": batch_acquisition[best].item()}]
        return Xopt, candidate_metadata

    def _augment_with_deriv_index(self, X: Tensor, indx):
        return torch.cat(
            (X, indx * torch.ones(X.shape[0], 1, dtype=self.dtype, device=self.device)),
            dim=1,
        )

    def _get_deriv_constraint_points(self):
        deriv_cp = torch.tensor([])
        for i in self.monotonic_idxs:
            induc_i = self._augment_with_deriv_index(self.inducing_points, i + 1)
            deriv_cp = torch.cat((deriv_cp, induc_i), dim=0)
        return deriv_cp

    def _get_acquisition_fn(self) -> MonotonicMCAcquisition:
        return self.acqf(
            model=self.model,
            deriv_constraint_points=self._get_deriv_constraint_points(),
            objective=self.objective,
            **self.extra_acqf_args,
        )


class MonotonicGPLSE(MonotonicRejectionGP):
    """
    Monotonic GP using the LSE acquisition function (with fixed beta for now, so it's
    the straddle) from:
    Gotovos et al., "Active learning for level set estimation", IJCAI 2013.

    In addition to the arguments in the parent class (MonotonicRejectionGP), requires
    specifying the target value for LSE.
    """

    def __init__(
        self,
        likelihood: str,
        monotonic_idxs: Sequence[int],
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        target_value: Optional[float] = None,
        num_induc: int = 25,
        num_samples: int = 250,
        num_rejection_samples: int = 5000,
        extra_acqf_args: Optional[dict[str, object]] = None,
    ) -> None:
        super().__init__(
            likelihood=likelihood,
            monotonic_idxs=monotonic_idxs,
            fixed_prior_mean=target_value,
            covar_module=covar_module,
            mean_module=mean_module,
            num_induc=num_induc,
            num_samples=num_samples,
            num_rejection_samples=num_rejection_samples,
            acqf=MonotonicMCLSE,
            objective=None,
            extra_acqf_args=extra_acqf_args,
        )
        self.target_value = target_value
        self.extra_acqf_args["target"] = target_value


class MonotonicGPLSETS(MonotonicGPLSE):
    """
    A monotonic GP with a Thompson-sampling-style approach for gen. We draw a
    posterior sample at a large number of points, and then choose the point
    that is closest to the target value.
    """

    def gen(
        self,
        n: int = 1,
        model_gen_options: Optional[Dict[str, Any]] = None,
        explore_features: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, Any]]]]:
        options = model_gen_options or {}
        num_ts_points = options.get("num_ts_points", 1000)  # OK for 2-d

        # Generate the points at which to sample
        X = draw_sobol_samples(bounds=self.bounds_, n=num_ts_points, q=1).squeeze(1)
        # Fix any explore features
        if explore_features is not None:
            for idx in explore_features:
                val = (
                    self.bounds_[0, idx]
                    + torch.rand(1, dtype=self.dtype)
                    * (self.bounds_[1, idx] - self.bounds_[0, idx])
                ).item()
                X[:, idx] = val

        # Draw n samples
        f_samp = self.sample(X, num_samples=n, num_rejection_samples=500)

        # Find the point closest to target
        dist = torch.abs(self.objective(f_samp) - self.target_value)
        best_indx = torch.argmin(dist, dim=1)
        return X[best_indx], {}


class MonotonicGPRand(MonotonicGPLSE):
    """
    A monotonic GP that uses random search for gen (as a baseline).
    """

    def gen(
        self,
        model_gen_options: Optional[Dict[str, object]] = None,
        explore_features: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, object]]]]:
        X = self.bounds_[0] + torch.rand(self.bounds_.shape[1]) * (
            self.bounds_[1] - self.bounds_[0]
        )
        return X.unsqueeze(0), {}
