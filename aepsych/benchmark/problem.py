#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import cached_property
from typing import Any, Dict, Union, List

import aepsych
import numpy as np
import torch
from scipy.stats import bernoulli, norm, pearsonr
from aepsych.strategy import SequentialStrategy, Strategy
from aepsych.utils import make_scaled_sobol


class Problem:
    """Wrapper for a problem or test function. Subclass from this
    and override f() to define your test function.
    """

    n_eval_points = 1000

    @cached_property
    def eval_grid(self):
        return make_scaled_sobol(lb=self.lb, ub=self.ub, size=self.n_eval_points)

    @property
    def name(self) -> str:
        raise NotImplementedError

    def f(self, x):
        raise NotImplementedError

    @cached_property
    def lb(self):
        return self.bounds[0]

    @cached_property
    def ub(self):
        return self.bounds[1]

    @property
    def bounds(self):
        raise NotImplementedError

    @property
    def metadata(self) -> Dict[str, Any]:
        """A dictionary of metadata passed to the Benchmark to be logged. Each key will become a column in the
        Benchmark's output dataframe, with its associated value stored in each row."""
        return {"name": self.name}

    def p(self, x: np.ndarray) -> np.ndarray:
        """Evaluate response probability from test function.

        Args:
            x (np.ndarray): Points at which to evaluate.

        Returns:
            np.ndarray: Response probability at queries points.
        """
        return norm.cdf(self.f(x))

    def sample_y(self, x: np.ndarray) -> np.ndarray:
        """Sample a response from test function.

        Args:
            x (np.ndarray): Points at which to sample.

        Returns:
            np.ndarray: A single (bernoulli) sample at points.
        """
        return bernoulli.rvs(self.p(x))

    def f_hat(self, model: aepsych.models.base.ModelProtocol) -> torch.Tensor:
        """Generate mean predictions from the model over the evaluation grid.

        Args:
            model (aepsych.models.base.ModelProtocol): Model to evaluate.

        Returns:
            torch.Tensor: Posterior mean from underlying model over the evaluation grid.
        """
        f_hat, _ = model.predict(self.eval_grid)
        return f_hat

    @cached_property
    def f_true(self) -> np.ndarray:
        """Evaluate true test function over evaluation grid.

        Returns:
            torch.Tensor: Values of true test function over evaluation grid.
        """
        return self.f(self.eval_grid).detach().numpy()

    @cached_property
    def p_true(self) -> torch.Tensor:
        """Evaluate true response probability over evaluation grid.

        Returns:
            torch.Tensor: Values of true response probability over evaluation grid.
        """
        return norm.cdf(self.f_true)

    def p_hat(self, model: aepsych.models.base.ModelProtocol) -> torch.Tensor:
        """Generate mean predictions from the model over the evaluation grid.

        Args:
            model (aepsych.models.base.ModelProtocol): Model to evaluate.

        Returns:
            torch.Tensor: Posterior mean from underlying model over the evaluation grid.
        """
        p_hat, _ = model.predict(self.eval_grid, probability_space=True)
        return p_hat

    def evaluate(
        self,
        strat: Union[Strategy, SequentialStrategy],
    ) -> Dict[str, float]:
        """Evaluate the strategy with respect to this problem.

        Extend this in subclasses to add additional metrics.
        Metrics include:
        - mae (mean absolute error), mae (mean absolute error), max_abs_err (max absolute error),
            pearson correlation. All of these are computed over the latent variable f and the
            outcome probability p, w.r.t. the posterior mean. Squared and absolute errors (miae, mise) are
            also computed in expectation over the posterior, by sampling.
        - Brier score, which measures how well-calibrated the outcome probability is, both at the posterior
            mean (plain brier) and in expectation over the posterior (expected_brier).

        Args:
            strat (aepsych.strategy.Strategy): Strategy to evaluate.

        Returns:
            Dict[str, float]: A dictionary containing metrics and their values.
        """
        # we just use model here but eval gets called on strat in case we need it in downstream evals
        # for example to separate out sobol vs opt trials
        model = strat.model
        assert model is not None, "Cannot evaluate strategy without a model!"

        # always eval f
        f_hat = self.f_hat(model).detach().numpy()
        p_hat = self.p_hat(model).detach().numpy()
        assert (
            self.f_true.shape == f_hat.shape
        ), f"self.f_true.shape=={self.f_true.shape} != f_hat.shape=={f_hat.shape}"

        mae_f = np.mean(np.abs(self.f_true - f_hat))
        mse_f = np.mean((self.f_true - f_hat) ** 2)
        max_abs_err_f = np.max(np.abs(self.f_true - f_hat))
        corr_f = pearsonr(self.f_true.flatten(), f_hat.flatten())[0]
        mae_p = np.mean(np.abs(self.p_true - p_hat))
        mse_p = np.mean((self.p_true - p_hat) ** 2)
        max_abs_err_p = np.max(np.abs(self.p_true - p_hat))
        corr_p = pearsonr(self.p_true.flatten(), p_hat.flatten())[0]
        brier = np.mean(2 * np.square(self.p_true - p_hat))

        # eval in samp-based expectation over posterior instead of just mean
        fsamps = model.sample(self.eval_grid, num_samples=1000).detach().numpy()
        try:
            psamps = (
                model.sample(self.eval_grid, num_samples=1000, probability_space=True)  # type: ignore
                .detach()
                .numpy()
            )
        except (
            TypeError
        ):  # vanilla models don't have proba_space samps, TODO maybe we should add them
            psamps = norm.cdf(fsamps)

        ferrs = fsamps - self.f_true[None, :]
        miae_f = np.mean(np.abs(ferrs))
        mise_f = np.mean(ferrs**2)

        perrs = psamps - self.p_true[None, :]
        miae_p = np.mean(np.abs(perrs))
        mise_p = np.mean(perrs**2)

        expected_brier = (2 * np.square(self.p_true[None, :] - psamps)).mean()

        metrics = {
            "mean_abs_err_f": mae_f,
            "mean_integrated_abs_err_f": miae_f,
            "mean_square_err_f": mse_f,
            "mean_integrated_square_err_f": mise_f,
            "max_abs_err_f": max_abs_err_f,
            "pearson_corr_f": corr_f,
            "mean_abs_err_p": mae_p,
            "mean_integrated_abs_err_p": miae_p,
            "mean_square_err_p": mse_p,
            "mean_integrated_square_err_p": mise_p,
            "max_abs_err_p": max_abs_err_p,
            "pearson_corr_p": corr_p,
            "brier": brier,
            "expected_brier": expected_brier,
        }

        return metrics


class LSEProblem(Problem):
    """Level set estimation problem.

    This extends the base problem class to evaluate the LSE/threshold estimate
    in addition to the function estimate.
    """

    def __init__(self, thresholds: Union[float, List]):
        super().__init__()
        thresholds = [thresholds] if isinstance(thresholds, float) else thresholds
        self.thresholds = np.array(thresholds)

        min_f = max(self.f_true.min(), -12)
        max_f = min(self.f_true.max(), 12)
        self.f_thresholds = np.arange(int(np.ceil(min_f)), int(np.floor(max_f))+1)

    @property
    def metadata(self) -> Dict[str, Any]:
        """A dictionary of metadata passed to the Benchmark to be logged. Each key will become a column in the
        Benchmark's output dataframe, with its associated value stored in each row."""
        md = super().metadata
        md["thresholds"] = (
            self.thresholds.tolist()
            if len(self.thresholds) > 1
            else self.thresholds.item()
        )
        return md

    def f_threshold(self, model=None):

        try:
            inverse_torch = model.likelihood.objective.inverse

            def inverse_link(x):
                return inverse_torch(torch.tensor(x)).numpy()

        except AttributeError:
            inverse_link = norm.ppf
        return inverse_link(self.thresholds).astype(np.float32)

    @cached_property
    def true_below_threshold(self) -> np.ndarray:
        """
        Evaluate whether the true function is below threshold over the eval grid
        (used for proper scoring and threshold missclassification metric).
        """
        return (
            self.p(self.eval_grid).reshape(1, -1) <= self.thresholds.reshape(-1, 1)
        ).astype(float)
    
    @cached_property
    def true_f_below_threshold(self) -> np.ndarray:
        return self.f_true.reshape(1, -1) < self.f_thresholds.reshape(-1, 1)


    def evaluate(self, strat: Union[Strategy, SequentialStrategy]) -> Dict[str, float]:
        """Evaluate the model with respect to this problem.

        For level set estimation, we add metrics w.r.t. the true threshold:
        - brier_p_below_{thresh), the brier score w.r.t. p(f(x)<thresh), in contrast to
            regular brier, which is the brier score for p(phi(f(x))=1), and the same
            for misclassification error.

        Args:
            strat (aepsych.strategy.Strategy): Strategy to evaluate.


        Returns:
            Dict[str, float]: A dictionary containing metrics and their values,
            including parent class metrics.
        """
        metrics = super().evaluate(strat)

        # we just use model here but eval gets called on strat in case we need it in downstream evals
        # for example to separate out sobol vs opt trials
        model = strat.model
        assert model is not None, "Cannot make predictions without a model!"

        # TODO bring back more threshold error metrics when we more clearly
        # define what "threshold" means in high-dim.

        # Brier score on level-set probabilities
        p_l = model.p_below_threshold(
            self.eval_grid, self.f_threshold(model)
        )
        true_p_l = self.true_below_threshold
        assert (
            p_l.ndim == 2
            and p_l.shape == true_p_l.shape
            and p_l.shape[0] == len(self.thresholds)
        )

        # Predict p(below threshold) at test points
        brier_p_below_thresh = np.mean(2 * np.square(true_p_l - p_l), axis=1)
        # Classification error in the probability space
        misclass_on_p_thresh = np.mean(
            p_l * (1 - true_p_l) + (1 - p_l) * true_p_l, axis=1
        )

        # classification error in units of latent space f(x) (or g(x1,x2))
        f, _ = model.predict(self.eval_grid)
        pred_f = f.numpy().reshape(1, -1) < self.f_thresholds.reshape(-1, 1)
        
        misclass_on_f_thresh = np.mean(
            pred_f * (1 - self.true_f_below_threshold) + (1 - pred_f) * self.true_f_below_threshold, axis=1
        )

        for i_threshold, threshold in enumerate(self.thresholds):
            metrics[f"brier_p_below_{threshold}"] = brier_p_below_thresh[i_threshold]
            metrics[f"misclass_on_thresh_{threshold}"] = misclass_on_p_thresh[i_threshold]

        for i_f_threshold, f_threshold in enumerate(self.f_thresholds):
            metrics[f"misclass_on_f_thresh_{int(f_threshold)}"] = misclass_on_f_thresh[i_f_threshold]
        return metrics


"""
The LSEProblemWithEdgeLogging class is copied from bernoulli_lse github repository
(https://github.com/facebookresearch/bernoulli_lse) by Letham et al. 2022.
"""


class LSEProblemWithEdgeLogging(LSEProblem):
    eps = 0.05

    def __init__(self, thresholds):
        super().__init__(thresholds)

    def evaluate(self, strat):
        metrics = super().evaluate(strat)

        # add number of edge samples to the log

        # get the trials selected by the final strat only
        n_opt_trials = strat.strat_list[-1].n_trials

        lb, ub = strat.lb, strat.ub
        r = ub - lb
        lb2 = lb + self.eps * r
        ub2 = ub - self.eps * r

        near_edge = (
            np.logical_or(
                (strat.x[-n_opt_trials:, :] <= lb2), (strat.x[-n_opt_trials:, :] >= ub2)
            )
            .any(axis=-1)
            .double()
        )

        metrics["prop_edge_sampling_mean"] = near_edge.mean().item()
        metrics["prop_edge_sampling_err"] = (
            2 * near_edge.std() / np.sqrt(len(near_edge))
        ).item()
        return metrics
