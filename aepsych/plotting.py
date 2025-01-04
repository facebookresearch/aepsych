#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sized,
    TypeAlias,
    Union,
)

import matplotlib

matplotlib.use("Agg")  # For headless plotting

import matplotlib.pyplot as plt
import numpy as np
import torch
from aepsych.strategy import Strategy
from aepsych.transforms import ParameterTransforms
from aepsych.transforms.ops import NormalizeScale
from aepsych.utils import dim_grid, get_lse_contour, get_lse_interval, make_scaled_sobol
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from scipy.stats import norm


linestyle_str = Literal["solid", "dashsed", "dashdot", "dotted"]
ColorType: TypeAlias = str


def plot_predict_1d(
    ax: Axes,
    x: torch.Tensor,
    prediction: torch.Tensor,
    pred_lower: Optional[torch.Tensor] = None,
    pred_upper: Optional[torch.Tensor] = None,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "",
    shaded_kwargs: Optional[Dict[Any, Any]] = None,
    **kwargs,
) -> Axes:
    """Return the ax with the model predictions plotted in place as a 1D line plot.
    Usually plots the predictions in the posterior space or the probability space.

    Args:
        ax (Axes): The Matplotlib axes to plot onto.
        x (torch.Tensor): The values where the model was evaluated, assumed to be
            ordered from lb to ub.
        prediction (torch.Tensor): The values of the predictions at every point it was
            evaluated, assumed to be the same order as x.
        post_lower (torch.Tensor, optional): The lower part of the shaded region around
            the prediction line, both post_lower/post_upper must be set to plot the band.
        post_upper (torch.Tensor, optional): The upper part of the shaded region around
            the prediction line, both post_lower/post_upper must be set to plot the band.
        x_label (str): The x axis label, defaults to "x".
        y_label (str): The y axis label, defaults to "y".
        title (str): The title of the plot, defaults to "".
        shaded_kwargs (Dict[Any, Any], optional): Kwargs to pass to the
            `ax.fill_between()` call to modify the shaded regions, defaults to None.
        **kwargs: Extra kwargs passed to the ax.plot() call, not passed to the plotting
            functions in charge of shaded regions.

    Returns:
        Axes: The input axes with the prediction plotted onto it. Note that plotting is
            done in-place.
    """
    ax.plot(x.squeeze(), prediction.squeeze(), **kwargs)

    if pred_lower is not None and pred_upper is not None:
        shaded_kwargs = shaded_kwargs or {}

        ax.fill_between(
            x.squeeze(),
            pred_lower.squeeze(),
            pred_upper.squeeze(),
            alpha=0.3,
            hatch="///",
            edgecolor="gray",
            **shaded_kwargs,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    return ax


def plot_points_1d(
    ax: Axes,
    x: torch.Tensor,
    y: torch.Tensor,
    pred_x: Optional[torch.Tensor] = None,
    pred_y: Optional[torch.Tensor] = None,
    point_size: float = 5.0,
    point_0_color: ColorType = "r",
    point_1_color: ColorType = "b",
    **kwargs,
) -> Axes:
    r"""Return the ax with the points plotted based on x and y in a 1D plot. If
    pred_x/pred_y is not set, these are plotted as marks at the bottom of the plot,
    otherwise, each point is plotted as close as possible to the line defined by
    the prediction values (pred_x/pred_y). Usually use alongside `plot_predict_1d()`.

    Args:
        ax (Axes): The Matplotlib axes to plot onto.
        x (torch.Tensor): The `(n, 1)` or `(n, d, 2)` points to plot. The 3D case will
            be considered a pairwise plot.
        y (torch.Tensor): The `(n, 1)` responses to plot.
        pred_x (torch.Tensor, optional): The points where the model was evaluated, used
            to position each point as close as possible to the line. If not set, the
            points are plotted as marks at the bottom of the plot.
        pred_y (torch.Tensor, optional): The model outputs at each point in pred_x, used to
            position each point as close as possible to the line. If not set, the points
            are plotted as marks at the bottom of the plot.
        point_size (float): The size of each plotted point, defaults to 5.0.
        point_0_color (ColorType): The color for the points where the response is 0. Uses
            Matplotlib's color types, defaults to "r".
        point_1_color (ColorType): The color for the points where the response is 1. Uses
            Matplotlib's color types, defaults to "b".
        **kwargs: Extra kwargs passed to the ax.plot() call, note that every point is
            plotted with an individual call so these kwargs must be applicable to single
            points.

    Returns:
        Axes: The input axes with the points plotted onto it. Note that plotting is done
            in-place.
    """
    if len(x.shape) == 3:  # Multi dim case
        if pred_x is not None and pred_y is not None:
            for pair, response in zip(x, y):
                x1 = pair[:, 0]
                x2 = pair[:, 1]
                y1 = pred_y[torch.argmin(torch.abs(x1 - pred_x))]
                y2 = pred_y[torch.argmin(torch.abs(x2 - pred_x))]

                ax.plot([x1, x2], [y1, y2], "-", c="gray", alpha=0.5)
                ax.plot(
                    x1,
                    y1,
                    marker="o",
                    color=point_1_color if response == 0 else point_0_color,
                    markersize=point_size,
                    alpha=0.5,
                    **kwargs,
                )
                ax.plot(
                    x2,
                    y2,
                    marker="o",
                    color=point_1_color if response == 1 else point_0_color,
                    markersize=point_size,
                    alpha=0.5,
                    **kwargs,
                )

        else:

            def curve(start, end, middle):
                poly_x = np.array((start[0], end[0], middle[0])).squeeze()
                poly_y = np.array((start[1], end[1], middle[1])).squeeze()
                f = np.poly1d(np.polyfit(poly_x, poly_y, 2))
                x = np.linspace(start[0], end[0], 100)
                return x, f(x)

            # Get where the hatches should be
            hatch_y, y_max = ax.get_ylim()
            mid_y = hatch_y + ((y_max - hatch_y) * 0.05)

            for pair, response in zip(x, y):
                x1 = pair[:, 0]
                x2 = pair[:, 1]

                # Create a curvey line between the hatches
                mid_x = torch.min(pair) + (torch.abs(x1 - x2) / 2)
                line_x, line_y = curve([x1, hatch_y], [x2, hatch_y], [mid_x, mid_y])
                ax.plot(line_x, line_y, "-", c="gray", alpha=0.5)
                ax.plot(
                    x1,
                    hatch_y,
                    marker=3,
                    color=point_1_color if response == 0 else point_0_color,
                    **kwargs,
                )
                ax.plot(
                    x2,
                    hatch_y,
                    marker=3,
                    color=point_1_color if response == 1 else point_0_color,
                    **kwargs,
                )
    else:
        if pred_x is not None and pred_y is not None:
            for x_, y_ in zip(x, y):
                plot_y = pred_y[torch.argmin(torch.abs(x_ - pred_x))]

                ax.plot(
                    x_,
                    plot_y,
                    marker="o",
                    color=point_1_color if y_ == 1 else point_0_color,
                    markersize=point_size,
                    alpha=0.5,
                    **kwargs,
                )
        else:
            # Get where the hatches should be
            hatch_y, y_max = ax.get_ylim()
            mid_y = hatch_y + ((y_max - hatch_y) * 0.05)

            for x_, y_ in zip(x, y):
                ax.plot(
                    x_,
                    hatch_y,
                    marker=3,
                    color=point_1_color if y_ == 1 else point_0_color,
                    **kwargs,
                )

    return ax


def plot_predict_2d(
    ax: Axes,
    prediction: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    edge_multiplier: float = 0.0,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "",
    colorbar: bool = True,
    **kwargs,
) -> Axes:
    """Return the ax with the model predictions plotted in-place as a 2D heatmap.
    Usually used to plot the model outputs in posterior space or probability space.

    Args:
        ax (Axes): The Matplotlib axes to plot onto.
        prediction (torch.Tensor): A 2D array of the predictions, assumes it was a square
            parameter grid where each cell was evaluated by the model.
        lb (torch.Tensor): The lower bound of the two parameters being plotted.
        ub (torch.Tensor): The upper bound of the two parameters being plotted.
        vmin (float, optional): The minimum value of the predictions, if not set, it will
            be the minimum of prediction.
        vmax (float, optional): The maximum of the predictions, if not set, it will be the
            maximum of prediction.
        edge_multiplier (float): How much to extend the plot extents beyond the parameter
            bounds (lb, ub), the plot extends beyond the bounds by the absolute difference
            multiplied by the edge_multiplier. Setting this to 0 will not extend the
            edge, any postive value will plot beyond the bounds, negative will plot less
            than the bounds. Used when you do not want the edges of the heatmap to be
            right at the parameter boundaries, especially useful if many points are at
            parameter boundaries. Defaults to 0.
        x_label (str): The x axis label, defaults to "x".
        y_label (str): The y axis label, defaults to "y".
        title (str): The title of this plot, defaults to "".
        colorbar (bool): Whether to add a colorbar (with bounds of [vmin, vmax]) to the
            parent figure of the Axes, defaults to True.
        **kwargs: Extra kwargs passed to the ax.imshow() call that creates the heatmap.

    Returns:
        Axes: The input axes with the predictions plotted onto it. Note that the plotting
            is done in-place.
    """
    # Make sure bounds are floats
    lb = lb.double()
    ub = ub.double()

    diff = np.abs(ub - lb)
    edge_bumps = (diff - (diff * (1 - edge_multiplier))) / 2
    lb -= edge_bumps
    ub += edge_bumps
    extent = (float(lb[0]), float(ub[0]), float(lb[1]), float(ub[1]))
    prediction = prediction.T
    prediction = torch.flip(prediction, dims=[0])
    mappable = ax.imshow(
        prediction,
        origin="upper",
        aspect=((ub[0] - lb[0]) / (ub[1] - lb[1])).item(),
        extent=extent,
        alpha=0.5,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )

    if colorbar and ax.figure is not None:
        # Orphaned axes will ignore colorbar, but it's rare
        ax.figure.colorbar(mappable)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return ax


def plot_points_2d(
    ax: Axes,
    x: torch.Tensor,
    y: torch.Tensor,
    point_size: float = 5.0,
    axis: Optional[List[int]] = None,
    slice_vals: Optional[torch.Tensor] = None,
    slice_gradient: float = 1.0,
    point_0_color: ColorType = "r",
    point_1_color: ColorType = "b",
    **kwargs,
) -> Axes:
    r"""Return the axes with the points defined by the parameters (x) and the outcomes
    (y) plotted in-place. If the inputs are from an experiment with stimuli per trial, a
    line is drawn between both. Usually used alongside `plot_predict_2d()`.

    Args:
        ax (Axes): The Matplotlib axes to plot onto.
        x (torch.Tensor): The `(n, d, 2)` or `(n, d)` data points to plot. Each point is
            a different trial.
        y (torch.Tensor): The `(n, 1)` responses given each set of datapoints. Each value
            is the response to a given trial.
        point_size (float): The size of the points, defaults to 5.
        axis (List[int], optional): If the dimensionality `d` is higher than 2, which
            two dimensions should the points be positioned with.
        slice_vals (torch.Tensor): If the dimensionality `d` is higher than 2, where was
            the other dimensions sliced at. This is used to determine the size/alpha
            gradient of the points based on how close each point is to the slice. If not
            set, the points will not change size/alpha based on slice distance. The
            Euclidean distance (s) of each point is calculated and converted to a
            multipler by `exp(-c * s)`, where c is the slice_gradient. The multipler is
            applied to the point_size and the alpha (transparency) of the points.
        slice_gradient (float): The rate at which the multiplier decreases as a function
            of the distance between a point and the slice. Defaults to 1.
        point_0_color (ColorType): The color for the points where the response is 0. Uses
            Matplotlib's color types, defaults to "r".
        point_1_color (ColorType): The color for the points where the response is 1. Uses
            Matplotlib's color types, defaults to "b".
        **kwargs: Extra kwargs passed to the ax.plot() call, note that every point is
            plotted with an individual call so these kwargs must be applicable to single
            points.

    Returns:
        Axes: The axes with the points plotted in-place.
    """
    if x.shape[1] > 2:
        if axis is None:
            raise ValueError(
                "x has more than 2 dimensions and no axis has been defined."
            )
        else:
            xcoords = x[:, axis[0], ...]
            ycoords = x[:, axis[1], ...]

            if slice_vals is not None:
                slice_vals = slice_vals.double()
                not_axis = [i for i in range(x.shape[1]) if i not in axis]
                normalize = ParameterTransforms(
                    normalize=NormalizeScale(d=x.shape[1], indices=not_axis)
                )
                not_x = normalize.transform(x)[:, not_axis, ...]

                # Freezes the inferred bounds
                normalize.eval()

                transformed_vals = torch.zeros(x.shape[1])
                transformed_vals[not_axis] = slice_vals
                transformed_vals = normalize.transform(transformed_vals)
                transformed_vals = transformed_vals[not_axis]

                slice_dist = not_x.sub(transformed_vals).pow(2).sum(dim=1).sqrt()

                # Calculate point alphas (which is also used as a size multiplier)
                point_alphas = torch.exp(-slice_dist * slice_gradient)
            else:
                point_alphas = torch.ones_like(xcoords)
    else:
        xcoords = x[:, 0]
        ycoords = x[:, 1]
        point_alphas = torch.ones_like(xcoords)

    if len(x.shape) == 3:  # n-wise experiment
        # Plot everything in pairs and add lines
        for xcoord, ycoord, y_, point_alpha in zip(xcoords, ycoords, y, point_alphas):
            ax.plot(xcoord, ycoord, "-", c="gray", alpha=0.5)
            ax.plot(
                xcoord[0],
                ycoord[0],
                marker="o",
                color=point_0_color if y_ == 0 else point_1_color,
                alpha=point_alpha[0].item(),
                markersize=point_alpha[0].item() * point_size,
                **kwargs,
            )
            ax.plot(
                xcoord[1],
                ycoord[1],
                marker="o",
                color=point_0_color if y_ == 1 else point_1_color,
                alpha=point_alpha[1].item(),
                markersize=point_alpha[1].item() * point_size,
                **kwargs,
            )
    else:
        for xcoord, ycoord, y_, point_alpha in zip(xcoords, ycoords, y, point_alphas):
            ax.plot(
                xcoord,
                ycoord,
                marker="o",
                color=point_0_color if y_ == 0 else point_1_color,
                alpha=point_alpha.item(),
                markersize=point_alpha.item() * point_size,
                **kwargs,
            )

    return ax


def plot_contours(
    ax: Axes,
    prediction: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    levels: Optional[torch.Tensor] = None,
    edge_multiplier: float = 0,
    color: Optional[ColorType] = "white",
    labels: bool = False,
    linestyles: Optional[linestyle_str] = "solid",
    **kwargs,
) -> Axes:
    """Plot contour lines at the levels onto the axes based on the model predictions
    with extents defined by lb and ub. Assumes that you're plotting ontop of a heatmap
    of those predictions given the same extents.

    Args:
        ax (Axes): The Matplotlib axes to plot onto.
        prediction (torch.Tensor): A 2D array of the predictions, assumes it was a square
            parameter grid where each cell was evaluated by the model.
        lb (torch.Tensor): The lower bound of the two parameters being plotted.
        ub (torch.Tensor): The upper bound of the two parameters being plotted.
        levels (torch.Tensor, optional): A sequence of values to plot the contours given
            the predictions. If not set, a contour will be plotted at each integer.
        edge_multiplier (float): How much to extend the plot extents beyond the parameter
            bounds (lb, ub), the plot extends beyond the bounds by the absolute difference
            multiplied by the edge_multiplier. Setting this to 0 will not extend the
            edge, any postive value will plot beyond the bounds, negative will plot less
            than the bounds. Used when you do not want the edges of the heatmap to be
            right at the parameter boundaries, especially useful if many points are at
            parameter boundaries. Defaults to 0.
        color (ColorType): What colors the contours should be, defaults to white.
        labels (bool): Whether or not to label the contours.
        linestyles (linestyle_str, optional): How should the contour lines be styled,
            defaults to "solid". Options are "solid", "dashsed", "dashdot", "dotted", can
            be set to None to default to the Matplotlib default.
        **kwargs: Extra keyword arguments to pass to the ax.contour() call that plots
            the contours.

    Returns:
        Axes: The axes with the points plotted in-place.
    """
    # Make sure bounds are floats
    lb = lb.double()
    ub = ub.double()

    diff = np.abs(ub - lb)
    edge_bumps = (diff - (diff * (1 - edge_multiplier))) / 2
    lb -= edge_bumps
    ub += edge_bumps
    extent = (float(lb[0]), float(ub[0]), float(lb[1]), float(ub[1]))

    prediction = prediction.T
    prediction = torch.flip(prediction, dims=[0])

    if levels is None:
        levels = torch.arange(
            prediction.min().floor().item(), prediction.max().ceil().item() + 1
        )

    contours = ax.contour(
        prediction,
        levels=levels,
        extent=extent,
        origin="upper",
        colors=color,
        linestyles=linestyles,
        **kwargs,
    )

    if labels:
        ax.clabel(contours, fontsize=10)

    return ax


def plot_strat(
    strat: Strategy,
    ax: Optional[plt.Axes] = None,
    true_testfun: Optional[Callable] = None,
    cred_level: float = 0.95,
    target_level: Optional[float] = 0.75,
    xlabel: Optional[str] = None,
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
        strat (Strategy): Strategy object to be plotted. Must have a dimensionality of 2 or less.
        ax (plt.Axes, optional): Matplotlib axis to plot on (if None, creates a new axis). Default: None.
        true_testfun (Callable, optional): Ground truth response function. Should take a n_samples x n_parameters tensor
                    as input and produce the response probability at each sample as output. Default: None.
        cred_level (float): Percentage of posterior mass around the mean to be shaded. Default: 0.95.
        target_level (float, optional): Response probability to estimate the threshold of. Default: 0.75.
        xlabel (str, optional): Label of the x-axis. Default: "Context (abstract)".
        ylabel (str, optional): Label of the y-axis (if None, defaults to "Response Probability" for 1-d plots or
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

    warnings.warn(
        "Plotting directly from strategy is deprecated, plots should be composed manually using the Matplotlib API, AEPsych specific helper functions are available in the plotting submodule.",
        DeprecationWarning,
    )
    assert (
        "binary" in strat.outcome_types
    ), f"Plotting not supported for outcome_type {strat.outcome_types[0]}"

    if target_level is not None and not hasattr(strat.model, "monotonic_idxs"):
        warnings.warn(
            "Threshold estimation may not be accurate for non-monotonic models."
        )

    if ax is None:
        _, ax = plt.subplots()

    if xlabel is None:
        xlabel = "Context (abstract)"

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

    elif dim == 3:
        raise RuntimeError("Use plot_strat_3d for 3d plots!")

    else:
        raise NotImplementedError("No plots for >3d!")

    ax.set_title(title)

    if include_legend:
        anchor = (1.4, 0.5) if include_colorbar and dim > 1 else (1, 0.5)
        plt.legend(loc="center left", bbox_to_anchor=anchor)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.tight_layout()

        if include_legend or (include_colorbar and dim > 1):
            plt.subplots_adjust(left=0.1, bottom=0.25, top=0.75)
        plt.show()


def _plot_strat_1d(
    strat: Strategy,
    ax: plt.Axes,
    true_testfun: Optional[Callable],
    cred_level: float,
    target_level: Optional[float],
    xlabel: str,
    ylabel: str,
    yes_label: str,
    no_label: str,
    gridsize: int,
) -> plt.Axes:
    """Helper function for creating 1-d plots. See plot_strat for an explanation of the arguments.

    Args:
        strat (Strategy): Strategy object to be plotted. Must have a dimensionality of 1.
        ax (plt.Axes): Matplotlib axis to plot on
        true_testfun (Callable, optional): Ground truth response function. Should take a n_samples x n_parameters tensor
                    as input and produce the response probability at each sample as output. Default: None.
        cred_level (float): Percentage of posterior mass around the mean to be shaded. Default: 0.95.
        target_level (float, optional): Response probability to estimate the threshold of. Default: 0.75.
        xlabel (str): Label of the x-axis. Default: "Context (abstract)".
        ylabel (str): Label of the y-axis (if None, defaults to "Response Probability" for 1-d plots or
                      "Intensity (Abstract)" for 2-d plots). Default: None.
        yes_label (str): Label of trials with response of 1. Default: "Yes trial".
        no_label (str): Label of trials with response of 0. Default: "No trial".
        gridsize (int): The number of points to sample each dimension at. Default: 30.

    Returns:
        plt.Axes: The axis object with the plot.
    """

    x, y = strat.x, strat.y
    assert x is not None and y is not None, "No data to plot!"

    if strat.model is not None:
        grid = dim_grid(lower=strat.lb, upper=strat.ub, gridsize=gridsize).cpu()
        samps = norm.cdf(strat.model.sample(grid, num_samples=10000).detach())
        phimean = samps.mean(0)
    else:
        raise RuntimeError("Cannot plot without a model!")

    ax.plot(np.squeeze(grid), phimean)
    if cred_level is not None:
        upper = np.quantile(samps, cred_level, axis=0)
        lower = np.quantile(samps, 1 - cred_level, axis=0)
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
            interpolate_monotonic(grid, s, target_level, strat.lb[0], strat.ub[0])
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
                grid,
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
        x[y == 0, 0],
        np.zeros_like(x[y == 0, 0]),
        marker="3",
        color="r",
        label=no_label,
    )
    ax.scatter(
        x[y == 1, 0],
        np.zeros_like(x[y == 1, 0]),
        marker="3",
        color="b",
        label=yes_label,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def _plot_strat_2d(
    strat: Strategy,
    ax: plt.Axes,
    true_testfun: Optional[Callable],
    cred_level: float,
    target_level: Optional[float],
    xlabel: str,
    ylabel: str,
    yes_label: str,
    no_label: str,
    flipx: bool,
    logx: bool,
    gridsize: int,
    include_colorbar: bool,
):
    """Helper function for creating 2-d plots. See plot_strat for an explanation of the arguments.

    Args:
        strat (Strategy): Strategy object to be plotted. Must have a dimensionality of 2.
        ax (plt.Axes): Matplotlib axis to plot on
        true_testfun (Callable, optional): Ground truth response function. Should take a n_samples x n_parameters tensor
                    as input and produce the response probability at each sample as output. Default: None.
        cred_level (float): Percentage of posterior mass around the mean to be shaded. Default: 0.95.
        target_level (float, optional): Response probability to estimate the threshold of. Default: 0.75.
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
        include_colorbar (bool): Whether to include the colorbar indicating the probability of "Yes" trials.
                                 Default: True.
    """

    x, y = strat.x, strat.y
    assert x is not None and y is not None, "No data to plot!"

    # make sure the model is fit well if we've been limiting fit time
    if strat.model is not None:
        strat.model.fit(train_x=x, train_y=y, max_fit_time=None)

        grid = dim_grid(lower=strat.lb, upper=strat.ub, gridsize=gridsize).cpu()
        fmean, _ = strat.model.predict(grid)
        phimean = norm.cdf(fmean.reshape(gridsize, gridsize).detach().numpy()).T
    else:
        raise RuntimeError("Cannot plot without a model!")

    lb = strat.transforms.untransform(strat.lb)
    ub = strat.transforms.untransform(strat.ub)

    extent = np.r_[lb[0], ub[0], lb[1], ub[1]]
    colormap = ax.imshow(
        phimean, aspect="auto", origin="lower", extent=extent, alpha=0.5
    )

    if flipx:
        extent = np.r_[lb[0], ub[0], ub[1], lb[1]]
        colormap = ax.imshow(
            phimean, aspect="auto", origin="upper", extent=extent, alpha=0.5
        )
    else:
        extent = np.r_[lb[0], ub[0], lb[1], ub[1]]
        colormap = ax.imshow(
            phimean, aspect="auto", origin="lower", extent=extent, alpha=0.5
        )

    # hacky relabel to be in logspace
    if logx:
        locs: np.ndarray = np.arange(lb[0], ub[0])
        ax.set_xticks(ticks=locs)
        ax.set_xticklabels(2.0**locs)

    ax.plot(x[y == 0, 0], x[y == 0, 1], "ro", alpha=0.7, label=no_label)
    ax.plot(x[y == 1, 0], x[y == 1, 1], "bo", alpha=0.7, label=yes_label)

    if target_level is not None:  # plot threshold
        mono_grid = np.linspace(lb[1], ub[1], num=gridsize)
        context_grid = np.linspace(lb[0], ub[0], num=gridsize)
        thresh_75, lower, upper = get_lse_interval(
            model=strat.model,
            mono_grid=mono_grid,
            target_level=target_level,
            grid_lb=strat.lb,
            grid_ub=strat.ub,
            cred_level=cred_level,
            mono_dim=1,
            lb=mono_grid.min(),
            ub=mono_grid.max(),
            gridsize=gridsize,
        )
        ax.plot(
            context_grid,
            thresh_75.cpu().numpy(),
            label=f"Est. {target_level*100:.0f}% threshold \n(with {cred_level*100:.0f}% posterior \nmass shaded)",
        )
        ax.fill_between(
            context_grid,
            lower.cpu().numpy(),
            upper.cpu().numpy(),
            alpha=0.3,
            hatch="///",
            edgecolor="gray",
        )

        if true_testfun is not None:
            true_f = true_testfun(grid).reshape(gridsize, gridsize)
            true_thresh = (
                get_lse_contour(
                    true_f,
                    mono_grid,
                    level=target_level,
                    lb=strat.lb[-1],
                    ub=strat.ub[-1],
                )
                .cpu()
                .numpy()
            )
            ax.plot(context_grid, true_thresh, label="Ground truth threshold")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if include_colorbar:
        colorbar = plt.colorbar(colormap, ax=ax)
        colorbar.set_label(f"Probability of {yes_label}")


def plot_strat_3d(
    strat: Strategy,
    parnames: Optional[List[str]] = None,
    outcome_label: str = "Yes Trial",
    slice_dim: int = 0,
    slice_vals: Union[List[float], int] = 5,
    contour_levels: Optional[Union[Iterable[float], bool]] = None,
    probability_space: bool = False,
    gridsize: int = 30,
    extent_multiplier: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Creates a plot of a 2d slice of a 3D strategy, showing the estimated model or probability response and contours
    Args:
        strat (Strategy): Strategy object to be plotted. Must have a dimensionality of 3.
        parnames (List[str], optional): list of the parameter names. If None, defaults to ["x1", "x2", "x3"].
        outcome_label (str): The label of the outcome variable
        slice_dim (int): dimension to slice on. Default: 0.
        slice_vals (Union[List[float], int]): values to take slices; OR number of values to take even slices from. Default: 5.
        contour_levels (Union[Iterable[float], bool], optional): List contour values to plot. Default: None. If true, all integer levels.
        probability_space (bool): Whether to plot probability. Default: False
        gridsize (int): The number of points to sample each dimension at. Default: 30.
        extent_multiplier (List[float], optional): multipliers for each of the dimensions when plotting. If None, defaults to [1, 1, 1].
        save_path (str, optional): File name to save the plot to. Default: None.
        show (bool): Whether the plot should be shown in an interactive window. Default: True.
    """
    warnings.warn(
        "Plotting directly from strategy is deprecated, plots should be composed manually using the Matplotlib API, AEPsych specific helper functions are available in the plotting submodule.",
        DeprecationWarning,
    )
    assert strat.model is not None, "Cannot plot without a model!"

    contour_levels_list: List[float] = []

    if parnames is None:
        parnames = ["x1", "x2", "x3"]
    # Get global min/max for all slices
    if probability_space:
        vmax = 1
        vmin = 0
        if contour_levels is True:
            contour_levels_list = [0.75]
    else:
        d = make_scaled_sobol(strat.lb, strat.ub, 2000)
        post = strat.model.posterior(d)
        fmean = post.mean.squeeze().detach().numpy()
        vmax = np.max(fmean)
        vmin = np.min(fmean)
        if contour_levels is True:
            contour_levels_list = list(np.arange(np.ceil(vmin), vmax + 1))

    if not isinstance(contour_levels_list, Sized):
        raise TypeError("contour_levels_list must be Sized (e.g., a list or an array).")

    # slice_vals is either a list of values or an integer number of values to slice on
    if isinstance(slice_vals, int):
        slices = np.linspace(strat.lb[slice_dim], strat.ub[slice_dim], slice_vals)
        slices = np.around(slices, 4)
    elif not isinstance(slice_vals, list):
        raise TypeError("slice_vals must be either an integer or a list of values")
    else:
        slices = np.array(slice_vals)

    # make mypy happy, note that this can't be more specific
    # because of https://github.com/numpy/numpy/issues/24738
    axs: np.ndarray[Any, Any]
    _, axs = plt.subplots(1, len(slices), constrained_layout=True, figsize=(20, 3))  # type: ignore

    assert len(slices) > 1, "Must have at least 2 slices"
    for _i, dim_val in enumerate(slices):
        img = plot_slice(
            axs[_i],
            strat,
            parnames,
            slice_dim,
            dim_val,
            vmin,
            vmax,
            gridsize,
            contour_levels_list,
            probability_space,
            extent_multiplier,
        )
    plt_parnames = np.delete(parnames, slice_dim)
    axs[0].set_ylabel(plt_parnames[1])
    cbar = plt.colorbar(img, ax=axs[-1])
    if probability_space:
        cbar.ax.set_ylabel(f"Probability of {outcome_label}")
    else:
        cbar.ax.set_ylabel(outcome_label)
    for clevel in contour_levels_list:  # type: ignore
        cbar.ax.axhline(y=clevel, c="w")

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()


def plot_slice(
    ax: Axes,
    strat: Strategy,
    parnames: List[str],
    slice_dim: int,
    slice_val: int,
    vmin: float,
    vmax: float,
    gridsize: int = 30,
    contour_levels: Optional[Sized] = None,
    lse: bool = False,
    extent_multiplier: Optional[List] = None,
) -> AxesImage:
    """Creates a plot of a 2d slice of a 3D strategy, showing the estimated model or probability response and contours
    Args:
        ax (plt.Axes): Matplotlib axis to plot on
        start (Strategy): Strategy object to be plotted. Must have a dimensionality of 3.
        parnames (List[str]): list of the parameter names.
        slice_dim (int): dimension to slice on.
        slice_val (int): value to take the slice along that dimension.
        vmin (float): global model minimum to use for plotting.
        vmax (float): global model maximum to use for plotting.
        gridsize (int): The number of points to sample each dimension at. Default: 30.
        contour_levels (Sized, optional): Contours to plot. Default: None
        lse (bool): Whether to plot probability. Default: False
        extent_multiplier (List, optional): multipliers for each of the dimensions when plotting. Default:None

    Returns:
        AxesImage: The axis object with the plot.
    """
    extent = np.c_[strat.lb, strat.ub].reshape(-1)
    if strat.model is not None:
        x = dim_grid(
            lower=strat.lb,
            upper=strat.ub,
            gridsize=gridsize,
            slice_dims={slice_dim: slice_val},
        ).cpu()
    else:
        raise RuntimeError("Cannot plot without a model!")
    if lse:
        fmean, fvar = strat.predict(x)
        fmean = fmean.detach().numpy().reshape(gridsize, gridsize)
        fmean = norm.cdf(fmean)
    else:
        post = strat.model.posterior(x)
        fmean = post.mean.squeeze().detach().numpy().reshape(gridsize, gridsize)

    # optionally rescale extents to correct values
    if extent_multiplier is not None:
        extent_scaled = extent * np.repeat(extent_multiplier, 2)
        dim_val_scaled = slice_val * extent_multiplier[slice_dim]
    else:
        extent_scaled = extent
        dim_val_scaled = slice_val

    plt_extents = np.delete(extent_scaled, [slice_dim * 2, slice_dim * 2 + 1])
    plt_parnames = np.delete(parnames, slice_dim)

    img = ax.imshow(
        fmean.T,
        extent=tuple(plt_extents),
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(parnames[slice_dim] + "=" + str(dim_val_scaled))
    ax.set_xlabel(plt_parnames[0])
    if contour_levels is not None:
        if len(contour_levels) > 0:
            ax.contour(
                fmean.T,
                contour_levels,
                colors="w",
                extent=plt_extents,
                origin="lower",
                aspect="auto",
            )
    else:
        raise (ValueError("Countour Levels should not be None!"))
    return img
