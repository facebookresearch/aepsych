#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from botorch.utils.sampling import draw_sobol_samples
from scipy.stats import norm

sns.set_theme()

from aepsych.config import Config
from aepsych.factory import (
    default_mean_covar_factory,
    song_mean_covar_factory,
    monotonic_mean_covar_factory,
)
from aepsych.models import GPClassificationModel, MonotonicRejectionGP
from aepsych.models.monotonic_rejection_gp import MixedDerivativeVariationalGP
from aepsych.utils import _dim_grid

global_seed = 3

def plot_prior_samps_1d():
    config = Config(
        config_dict={
            "common": {
                "outcome_type": "single_probit",
                "target": 0.75,
                "lb": "[-3]",
                "ub": "[3]",
            },
            "default_mean_covar_factory": {},
            "song_mean_covar_factory": {},
            "monotonic_mean_covar_factory": {"monotonic_idxs": "[0]"},
        }
    )
    lb = torch.Tensor([-3])
    ub = torch.Tensor([3])
    nsamps = 10
    gridsize = 50
    grid = _dim_grid(lower=lb, upper=ub, dim=1, gridsize=gridsize)
    np.random.seed(global_seed)
    torch.random.manual_seed(global_seed)
    with gpytorch.settings.prior_mode(True):
        rbf_mean, rbf_covar = default_mean_covar_factory(config)
        rbf_model = GPClassificationModel(
            inducing_min=lb,
            inducing_max=ub,
            inducing_size=100,
            mean_module=rbf_mean,
            covar_module=rbf_covar,
        )
        # add just two samples at high and low
        rbf_model.set_train_data(
            torch.Tensor([-3, 3])[:, None], torch.LongTensor([0, 1])
        )
        rbf_samps = rbf_model(grid).sample(torch.Size([nsamps]))

        song_mean, song_covar = song_mean_covar_factory(config)
        song_model = GPClassificationModel(
            inducing_min=lb,
            inducing_max=ub,
            inducing_size=100,
            mean_module=song_mean,
            covar_module=song_covar,
        )
        song_model.set_train_data(
            torch.Tensor([-3, 3])[:, None], torch.LongTensor([0, 1])
        )

        song_samps = song_model(grid).sample(torch.Size([nsamps]))

        mono_mean, mono_covar = monotonic_mean_covar_factory(config)
        mono_model = MonotonicRejectionGP(
            likelihood="probit-bernoulli",
            monotonic_idxs=[0],
            mean_module=mono_mean,
            covar_module=mono_covar,
        )

        bounds_ = torch.tensor([-3.0, 3.0])[:, None]
        # Select inducing points
        mono_model.inducing_points = draw_sobol_samples(
            bounds=bounds_, n=mono_model.num_induc, q=1
        ).squeeze(1)

        inducing_points_aug = mono_model._augment_with_deriv_index(
            mono_model.inducing_points, 0
        )
        scales = ub - lb
        dummy_train_x = mono_model._augment_with_deriv_index(
            torch.Tensor([-3, 3])[:, None], 0
        )
        mono_model.model = MixedDerivativeVariationalGP(
            train_x=dummy_train_x,
            train_y=torch.LongTensor([0, 1]),
            inducing_points=inducing_points_aug,
            scales=scales,
            fixed_prior_mean=torch.Tensor([0.75]),
            covar_module=mono_covar,
            mean_module=mono_mean,
        )
        mono_samps = mono_model.sample(grid, nsamps)

    fig, ax = plt.subplots(1, 3, figsize=(7.5, 3))
    fig.tight_layout(rect=[0.01, 0.03, 1, 0.9])
    fig.suptitle("GP prior samples (probit-transformed)")
    ax[0].plot(grid.squeeze(), norm.cdf(song_samps.T), "b")
    ax[0].set_ylabel("Response Probability")
    ax[0].set_title("Linear kernel")

    ax[1].plot(grid.squeeze(), norm.cdf(rbf_samps.T), "b")
    ax[1].set_xlabel("Intensity")
    ax[1].set_title("RBF kernel (nonmonotonic)")

    ax[2].plot(grid.squeeze(), norm.cdf(mono_samps.T), "b")
    ax[2].set_title("RBF kernel (monotonic)")
    return fig


def plot_prior_samps_2d():
    config = Config(
        config_dict={
            "common": {
                "outcome_type": "single_probit",
                "target": 0.75,
                "lb": "[-3, -3]",
                "ub": "[3, 3]",
            },
            "default_mean_covar_factory": {},
            "song_mean_covar_factory": {},
            "monotonic_mean_covar_factory": {"monotonic_idxs": "[1]"},
        }
    )
    lb = torch.Tensor([-3, -3])
    ub = torch.Tensor([3, 3])
    nsamps = 5
    gridsize = 30
    grid = _dim_grid(lower=lb, upper=ub, dim=2, gridsize=gridsize)
    np.random.seed(global_seed)
    torch.random.manual_seed(global_seed)
    with gpytorch.settings.prior_mode(True):
        rbf_mean, rbf_covar = default_mean_covar_factory(config)
        rbf_model = GPClassificationModel(
            inducing_min=lb,
            inducing_max=ub,
            inducing_size=100,
            mean_module=rbf_mean,
            covar_module=rbf_covar,
        )
        # add just two samples at high and low
        rbf_model.set_train_data(torch.Tensor([-3, -3])[:, None], torch.LongTensor([0]))
        rbf_samps = rbf_model(grid).sample(torch.Size([nsamps]))

        song_mean, song_covar = song_mean_covar_factory(config)
        song_model = GPClassificationModel(
            inducing_min=lb,
            inducing_max=ub,
            inducing_size=100,
            mean_module=song_mean,
            covar_module=song_covar,
        )
        song_model.set_train_data(
            torch.Tensor([-3, -3])[:, None], torch.LongTensor([0])
        )

        song_samps = song_model(grid).sample(torch.Size([nsamps]))

        mono_mean, mono_covar = monotonic_mean_covar_factory(config)
        mono_model = MonotonicRejectionGP(
            likelihood="probit-bernoulli",
            monotonic_idxs=[1],
            mean_module=mono_mean,
            covar_module=mono_covar,
            num_induc=1000,
        )

        bounds_ = torch.tensor([-3.0, -3.0, 3.0, 3.0]).reshape(2, -1)
        # Select inducing points
        mono_model.inducing_points = draw_sobol_samples(
            bounds=bounds_, n=mono_model.num_induc, q=1
        ).squeeze(1)

        inducing_points_aug = mono_model._augment_with_deriv_index(
            mono_model.inducing_points, 0
        )
        scales = ub - lb
        dummy_train_x = mono_model._augment_with_deriv_index(
            torch.Tensor([-3, 3])[None, :], 0
        )
        mono_model.model = MixedDerivativeVariationalGP(
            train_x=dummy_train_x,
            train_y=torch.LongTensor([0]),
            inducing_points=inducing_points_aug,
            scales=scales,
            fixed_prior_mean=torch.Tensor([0.75]),
            covar_module=mono_covar,
            mean_module=mono_mean,
        )
        mono_samps = mono_model.sample(grid, nsamps)

    intensity_grid = np.linspace(-3, 3, gridsize)
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 3))
    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    fig.suptitle("Prior samples")

    square_samps = np.array([s.reshape((gridsize,) * 2).numpy() for s in song_samps])
    plotsamps = norm.cdf(square_samps[:, ::5, :]).T.reshape(gridsize, -1)
    ax[0].plot(intensity_grid, plotsamps, "b")
    ax[0].set_title("Linear kernel model")

    square_samps = np.array([s.reshape((gridsize,) * 2).numpy() for s in rbf_samps])
    plotsamps = norm.cdf(square_samps[:, ::5, :]).T.reshape(gridsize, -1)
    ax[1].plot(intensity_grid, plotsamps, "b")
    ax[1].set_title("Nonmonotonic RBF kernel model")

    square_samps = np.array([s.reshape((gridsize,) * 2).numpy() for s in mono_samps])
    plotsamps = norm.cdf(square_samps[:, ::5, :]).T.reshape(gridsize, -1)
    ax[2].plot(intensity_grid, plotsamps, "b")
    ax[2].set_title("Monotonic RBF kernel model")

    return fig


if __name__ == "__main__":

    prior_samps_1d = plot_prior_samps_1d()
    prior_samps_1d.savefig("./figs/prior_samps.pdf", dpi=200)
