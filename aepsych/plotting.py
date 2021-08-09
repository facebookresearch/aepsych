#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional
import matplotlib.pyplot as plt
import numpy as np
from aepsych.utils import get_lse_interval, _dim_grid, get_lse_contour
from aepsych.strategy import ModelWrapperStrategy
from scipy.stats import norm


def plot_strat(
    strat: ModelWrapperStrategy,
    ax: Optional[plt.Axes] = None,
    true_testfun: Optional[Callable] = None,
    cred_level: float = 0.95,
    target_level: float = 0.75,
    xlabel: str = "Context (abstract)",
    ylabel: Optional[str] = None,
    yes_label: str = "Yes trial",
    no_label: str = "No trial",
    flipx: bool = False,
    logx: bool = False,
    gridsize: int = 30,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
    include_legend: bool = True,
    include_colorbar: bool = True,
) -> None:
    """Creates a plot of a strategy, showing participants responses on each trial, the estimated response function and
    threshold, and optionally a ground truth response threshold.

    Args:
        strat (ModelWrapperStrategy): Strategy object to be plotted. Must have a dimensionality of 2 or less.
        ax (plt.Axes, optional): Matplotlib axis to plot on (if None, creates a new axis). Default: None.
        true_testfun (Callable, optional): Ground truth response function. Should take a n_samples x n_parameters tensor 
                    as input and produce the response probability at each sample as output. Default: None.
        cred_level (float): Percentage of posterior mass around the mean to be shaded. Default: 0.95.
        target_level (float): Response probability to estimate the threshold of. Default: 0.75.
        xlabel (str): Label of the x-axis. Default: "Context (abstract)".
        ylabel (str): Label of the y-axis (if None, defaults to "Response Probability" for 1-d plots or 
                      "Intensity (Abstract)" for 2-d plots). Default: None.
        yes_label (str): Label of trials with response of 1. Default: "Yes trial".
        no_label (str): Label of trials with response of 0. Default: "No trial".
        flipx (bool): Whether the values of the x-axis should be flipped such that the min becomes the max and vice 
                      versa.
               (Only valid for 2-d plots.) Default: False.
        logx (bool): Whether the x-axis should be log-transformed. (Only valid for 2-d plots.) Default: False.
        gridsize (int): The number of points to sample each dimension at. Default: 30.
        title (str): Title of the plot. Default: ''.
        save_path (str, optional): File name to save the plot to. Default: None.
        show (bool): Whether the plot should be shown in an interactive window. Default: True.
        include_legend (bool): Whether to include the legend in the figure. Default: True.
        include_colorbar (bool): Whether to include the colorbar indicating the probability of "Yes" trials. 
                                 Default: True.
    """
    if ax is None:
        _, ax = plt.subplots()

    dim = strat.dim
    if dim == 1:
        if ylabel is None:
            ylabel = "Response Probability"
        _plot_strat_1d(
            strat,
            ax,
            true_testfun,
            cred_level,
            target_level,
            xlabel,
            ylabel,
            yes_label,
            no_label,
            gridsize,
        )

    elif dim == 2:
        if ylabel is None:
            ylabel = "Intensity (abstract)"
        _plot_strat_2d(
            strat,
            ax,
            true_testfun,
            cred_level,
            target_level,
            xlabel,
            ylabel,
            yes_label,
            no_label,
            flipx,
            logx,
            gridsize,
            include_colorbar,
        )

    # TODO implement 3d plots
    else:
        raise NotImplementedError("No plots for >2d")

    ax.set_title(title)

    if include_legend:
        plt.legend(loc="center left", bbox_to_anchor=(1.4, 0.5))

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.tight_layout()
        plt.show()


def _plot_strat_1d(
    strat: ModelWrapperStrategy,
    ax: plt.Axes,
    true_testfun: Optional[Callable],
    cred_level: float,
    target_level: float,
    xlabel: str,
    ylabel: str,
    yes_label: str,
    no_label: str,
    gridsize: int,
):
    """Helper function for creating 1-d plots. See plot_strat for an explanation of the arguments."""

    x, y = strat.x, strat.y

    grid = _dim_grid(modelbridge=strat.modelbridge, gridsize=gridsize)
    samps = norm.cdf(strat.modelbridge.sample(grid))
    phimean = samps.mean(0)
    upper = np.quantile(samps, cred_level, axis=0)
    lower = np.quantile(samps, 1 - cred_level, axis=0)

    ax.plot(np.squeeze(grid), phimean)
    ax.fill_between(
        np.squeeze(grid),
        lower,
        upper,
        alpha=0.3,
        hatch="///",
        edgecolor="gray",
        label=f"{cred_level*100:.0f}% posterior mass",
    )
    if target_level is not None:
        from aepsych.utils import interpolate_monotonic

        threshold_samps = [
            interpolate_monotonic(
                grid.squeeze().numpy(), s, target_level, strat.lb[0], strat.ub[0]
            )
            for s in samps
        ]
        thresh_med = np.mean(threshold_samps)
        thresh_lower = np.quantile(threshold_samps, q=1 - cred_level)
        thresh_upper = np.quantile(threshold_samps, q=cred_level)

        ax.errorbar(
            thresh_med,
            target_level,
            xerr=np.r_[thresh_med - thresh_lower, thresh_upper - thresh_med][:, None],
            capsize=5,
            elinewidth=1,
            label=f"Est. {target_level*100:.0f}% threshold \n(with {cred_level*100:.0f}% posterior \nmass marked)",
        )

    if true_testfun is not None:
        true_f = true_testfun(grid)
        ax.plot(grid, true_f.squeeze(), label="True function")
        if target_level is not None:
            true_thresh = interpolate_monotonic(
                grid.squeeze().numpy(),
                true_f.squeeze(),
                target_level,
                strat.lb[0],
                strat.ub[0],
            )

            ax.plot(
                true_thresh,
                target_level,
                "o",
                label=f"True {target_level*100:.0f}% threshold",
            )

    ax.scatter(
        x[y == 0, 0], np.zeros_like(x[y == 0, 0]), marker=3, color="r", label=no_label,
    )
    ax.scatter(
        x[y == 1, 0], np.zeros_like(x[y == 1, 0]), marker=3, color="b", label=yes_label,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def _plot_strat_2d(
    strat: ModelWrapperStrategy,
    ax: plt.Axes,
    true_testfun: Optional[Callable],
    cred_level: float,
    target_level: float,
    xlabel: str,
    ylabel: str,
    yes_label: str,
    no_label: str,
    flipx: bool,
    logx: bool,
    gridsize: int,
    include_colorbar: bool,
):
    """Helper function for creating 2-d plots. See plot_strat for an explanation of the arguments."""

    x, y = strat.x, strat.y

    grid = _dim_grid(modelbridge=strat.modelbridge, gridsize=gridsize)
    fmean, _ = strat.modelbridge.predict(grid)
    phimean = norm.cdf(fmean.reshape(gridsize, gridsize).detach().numpy()).T

    extent = np.r_[strat.lb[0], strat.ub[0], strat.lb[1], strat.ub[1]]
    colormap = ax.imshow(
        phimean, aspect="auto", origin="lower", extent=extent, alpha=0.5
    )

    if flipx:
        extent = np.r_[strat.lb[0], strat.ub[0], strat.ub[1], strat.lb[1]]
        colormap = ax.imshow(
            phimean, aspect="auto", origin="upper", extent=extent, alpha=0.5
        )
    else:
        extent = np.r_[strat.lb[0], strat.ub[0], strat.lb[1], strat.ub[1]]
        colormap = ax.imshow(
            phimean, aspect="auto", origin="lower", extent=extent, alpha=0.5
        )

    # hacky relabel to be in logspace
    if logx:
        locs = np.arange(strat.lb[0], strat.ub[0])
        ax.set_xticks(ticks=locs)
        ax.set_xticklabels(2.0 ** locs)

    ax.plot(x[y == 0, 0], x[y == 0, 1], "ro", alpha=0.7, label=no_label)
    ax.plot(x[y == 1, 0], x[y == 1, 1], "bo", alpha=0.7, label=yes_label)

    if target_level is not None:  # plot threshold
        mono_grid = np.linspace(strat.lb[1], strat.ub[1], num=gridsize)
        context_grid = np.linspace(strat.lb[0], strat.ub[0], num=gridsize)
        thresh_75, lower, upper = get_lse_interval(
            modelbridge=strat.modelbridge,
            mono_grid=mono_grid,
            target_level=target_level,
            cred_level=cred_level,
            mono_dim=1,
            lb=mono_grid.min(),
            ub=mono_grid.max(),
            gridsize=gridsize,
        )
        ax.plot(
            context_grid,
            thresh_75,
            label=f"Est. {target_level*100:.0f}% threshold \n(with {cred_level*100:.0f}% posterior \nmass shaded)",
        )
        ax.fill_between(
            context_grid, lower, upper, alpha=0.3, hatch="///", edgecolor="gray"
        )

        if true_testfun is not None:
            true_f = true_testfun(grid).reshape(gridsize, gridsize)
            true_thresh = get_lse_contour(
                true_f, mono_grid, level=target_level, lb=strat.lb[-1], ub=strat.ub[-1]
            )
            ax.plot(context_grid, true_thresh, label="Ground truth threshold")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if include_colorbar:
        colorbar = plt.colorbar(colormap, ax=ax)
        colorbar.set_label(f"Probability of {yes_label}")
