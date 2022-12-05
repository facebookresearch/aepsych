#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple, Union

import gpytorch
import numpy as np
import torch
from aepsych.config import Config
from aepsych.factory.factory import default_mean_covar_factory
from aepsych.models.base import AEPsychMixin
from aepsych.models.utils import select_inducing_points
from aepsych.utils import _process_bounds, promote_0d
from aepsych.utils_logging import getLogger
from gpytorch.likelihoods import BernoulliLikelihood, Likelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from scipy.special import owens_t
from scipy.stats import norm
from torch.distributions import Normal

logger = getLogger()


class GPClassificationModel(AEPsychMixin, ApproximateGP):
    """Probit-GP model with variational inference.

    From a conventional ML perspective this is a GP Classification model,
    though in the psychophysics context it can also be thought of as a
    nonlinear generalization of the standard linear model for 1AFC or
    yes/no trials.

    For more on variational inference, see e.g.
    https://docs.gpytorch.ai/en/v1.1.1/examples/04_Variational_and_Approximate_GPs/
    """

    _batch_size = 1
    _num_outputs = 1
    stimuli_per_trial = 1
    outcome_type = "binary"

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        dim: Optional[int] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        inducing_size: int = 100,
        max_fit_time: Optional[float] = None,
        inducing_point_method: str = "auto",
    ):
        """Initialize the GP Classification model

        Args:
            lb (Union[numpy.ndarray, torch.Tensor]): Lower bounds of the parameters.
            ub (Union[numpy.ndarray, torch.Tensor]): Upper bounds of the parameters.
            dim (int, optional): The number of dimensions in the parameter space. If None, it is inferred from the size
                of lb and ub.
            mean_module (gpytorch.means.Mean, optional): GP mean class. Defaults to a constant with a normal prior.
            covar_module (gpytorch.kernels.Kernel, optional): GP covariance kernel class. Defaults to scaled RBF with a
                gamma prior.
            likelihood (gpytorch.likelihood.Likelihood, optional): The likelihood function to use. If None defaults to
                Bernouli likelihood.
            inducing_size (int): Number of inducing points. Defaults to 100.
            max_fit_time (float, optional): The maximum amount of time, in seconds, to spend fitting the model. If None,
                there is no limit to the fitting time.
            inducing_point_method (string): The method to use to select the inducing points. Defaults to "auto".
                If "sobol", a number of Sobol points equal to inducing_size will be selected.
                If "pivoted_chol", selects points based on the pivoted Cholesky heuristic.
                If "kmeans++", selects points by performing kmeans++ clustering on the training data.
                If "auto", tries to determine the best method automatically.
        """
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        self.max_fit_time = max_fit_time
        self.inducing_size = inducing_size

        if likelihood is None:
            likelihood = BernoulliLikelihood()

        self.inducing_point_method = inducing_point_method

        # initialize to sobol before we have data
        inducing_points = select_inducing_points(
            inducing_size=self.inducing_size, bounds=self.bounds, method="sobol"
        )

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=torch.Size([self._batch_size])
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        )
        super().__init__(variational_strategy)

        if mean_module is None or covar_module is None:
            config = Config(
                config_dict={
                    "default_mean_covar_factory": {
                        "lb": str(self.lb.tolist()),
                        "ub": str(self.ub.tolist()),
                    }
                }
            )  # type: ignore
            default_mean, default_covar = default_mean_covar_factory(config)

        self.mean_module = mean_module or default_mean
        self.covar_module = covar_module or default_covar
        self.likelihood = likelihood

        self._fresh_state_dict = deepcopy(self.state_dict())
        self._fresh_likelihood_dict = deepcopy(self.likelihood.state_dict())

    @classmethod
    def from_config(cls, config: Config) -> GPClassificationModel:
        """Alternate constructor for GPClassification model.

        This is used when we recursively build a full sampling strategy
        from a configuration. TODO: document how this works in some tutorial.

        Args:
            config (Config): A configuration containing keys/values matching this class

        Returns:
            GPClassificationModel: Configured class instance.
        """

        classname = cls.__name__
        inducing_size = config.getint(classname, "inducing_size", fallback=10)

        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        dim = config.getint(classname, "dim", fallback=None)

        mean_covar_factory = config.getobj(
            classname, "mean_covar_factory", fallback=default_mean_covar_factory
        )

        mean, covar = mean_covar_factory(config)
        max_fit_time = config.getfloat(classname, "max_fit_time", fallback=None)

        inducing_point_method = config.get(
            classname, "inducing_point_method", fallback="auto"
        )

        likelihood_cls = config.getobj(classname, "likelihood", fallback=None)

        if likelihood_cls is not None:
            if hasattr(likelihood_cls, "from_config"):
                likelihood = likelihood_cls.from_config(config)
            else:
                likelihood = likelihood_cls()
        else:
            likelihood = None  # fall back to __init__ default

        return cls(
            lb=lb,
            ub=ub,
            dim=dim,
            inducing_size=inducing_size,
            mean_module=mean,
            covar_module=covar,
            max_fit_time=max_fit_time,
            inducing_point_method=inducing_point_method,
            likelihood=likelihood,
        )

    def _reset_hyperparameters(self):
        # warmstart_hyperparams affects hyperparams but not the variational strat,
        # so we keep the old variational strat (which is only refreshed
        # if warmstart_induc=False).
        vsd = self.variational_strategy.state_dict()  # type: ignore
        vsd_hack = {f"variational_strategy.{k}": v for k, v in vsd.items()}
        state_dict = deepcopy(self._fresh_state_dict)
        state_dict.update(vsd_hack)
        self.load_state_dict(state_dict)
        self.likelihood.load_state_dict(self._fresh_likelihood_dict)

    def _reset_variational_strategy(self):
        inducing_points = select_inducing_points(
            inducing_size=self.inducing_size,
            model=self,
            X=self.train_inputs[0],
            bounds=self.bounds,
            method=self.inducing_point_method,
        )
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=torch.Size([self._batch_size])
        )
        self.variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        )

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        warmstart_hyperparams: bool = False,
        warmstart_induc: bool = False,
        **kwargs,
    ) -> None:
        """Fit underlying model.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.LongTensor): Responses.
            warmstart_hyperparams (bool): Whether to reuse the previous hyperparameters (True) or fit from scratch
                (False). Defaults to False.
            warmstart_induc (bool): Whether to reuse the previous inducing points or fit from scratch (False).
                Defaults to False.
        """
        self.set_train_data(train_x, train_y)

        # by default we reuse the model state and likelihood. If we
        # want a fresh fit (no warm start), copy the state from class initialization.
        if not warmstart_hyperparams:
            self._reset_hyperparameters()

        if not warmstart_induc:
            self._reset_variational_strategy()

        n = train_y.shape[0]
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, n)

        self._fit_mll(mll, **kwargs)

    def sample(
        self, x: Union[torch.Tensor, np.ndarray], num_samples: int
    ) -> torch.Tensor:
        """Sample from underlying model.

        Args:
            x (torch.Tensor): Points at which to sample.
            num_samples (int, optional): Number of samples to return. Defaults to None.
            kwargs are ignored

        Returns:
            torch.Tensor: Posterior samples [num_samples x dim]
        """
        return self.posterior(x).rsample(torch.Size([num_samples])).detach().squeeze()

    def predict(
        self, x: Union[torch.Tensor, np.ndarray], probability_space: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            probability_space (bool, optional): Return outputs in units of
                response probability instead of latent function value. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Posterior mean and variance at queries points.
        """
        with torch.no_grad():
            post = self.posterior(x)
        fmean = post.mean.squeeze()
        fvar = post.variance.squeeze()
        if probability_space:
            if isinstance(self.likelihood, BernoulliLikelihood):
                # Probability-space mean and variance for Bernoulli-probit models is
                # available in closed form, Proposition 1 in Letham et al. 2022 (AISTATS).
                a_star = fmean / torch.sqrt(1 + fvar)
                pmean = Normal(0, 1).cdf(a_star)
                t_term = torch.tensor(
                    owens_t(a_star.numpy(), 1 / np.sqrt(1 + 2 * fvar.numpy())),
                    dtype=a_star.dtype,
                )
                pvar = pmean - 2 * t_term - pmean.square()
                return promote_0d(pmean), promote_0d(pvar)
            else:
                fsamps = post.sample(torch.Size([10000]))
                if hasattr(self.likelihood, "objective"):
                    psamps = self.likelihood.objective(fsamps)
                else:
                    psamps = norm.cdf(fsamps)
                pmean, pvar = psamps.mean(0), psamps.var(0)
                return promote_0d(pmean), promote_0d(pvar)

        else:
            return promote_0d(fmean), promote_0d(fvar)

    def update(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs):
        """Perform a warm-start update of the model from previous fit."""
        return self.fit(
            train_x, train_y, warmstart_hyperparams=True, warmstart_induc=True, **kwargs
        )
