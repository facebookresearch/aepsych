#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, TypeAlias, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from aepsych.transforms import ParameterTransforms
from aepsych.transforms.ops import NormalizeScale
from matplotlib.axes import Axes

linestyle_str = Literal["solid", "dashsed", "dashdot", "dotted"]
ColorType: TypeAlias = str


def plot_predict_1d(
    x: Iterable[float],
    prediction: Iterable[float],
    ax: Optional[Axes] = None,
    pred_lower: Optional[Iterable[float]] = None,
    pred_upper: Optional[Iterable[float]] = None,
    shaded_kwargs: Optional[Dict[Any, Any]] = None,
    **kwargs,
) -> Axes:
    """Return the ax with the model predictions plotted in place as a 1D line plot.
    Usually plots the predictions in the posterior space or the probability space.

    Args:
        x (Iterable[float]): The values where the model was evaluated, assumed to be
            ordered from lb to ub.
        prediction (Iterable[float]): The values of the predictions at every point it was
            evaluated, assumed to be the same order as x.
        ax (Axes, optional): The Matplotlib axes to plot onto. If not set, an axes is
            made and returned.
        post_lower (Iterable[float], optional): The lower part of the shaded region around
            the prediction line, both post_lower/post_upper must be set to plot the band.
        post_upper (Iterable[float], optional): The upper part of the shaded region around
            the prediction line, both post_lower/post_upper must be set to plot the band.
        shaded_kwargs (Dict[Any, Any], optional): Kwargs to pass to the
            `ax.fill_between()` call to modify the shaded regions, defaults to None.
        **kwargs: Extra kwargs passed to the ax.plot() call, not passed to the plotting
            functions in charge of shaded regions.

    Returns:
        Axes: The input axes with the prediction plotted onto it. Note that plotting is
            done in-place.
    """
    x, prediction, pred_lower, pred_upper = _tensor_cast(
        x, prediction, pred_lower, pred_upper
    )
    if ax is None:
        _, ax = plt.subplots()

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

    return ax


def plot_points_1d(
    x: Iterable[Union[float, Iterable[float]]],
    y: Iterable[float],
    ax: Optional[Axes] = None,
    pred_x: Optional[Iterable[float]] = None,
    pred_y: Optional[Iterable[float]] = None,
    point_size: float = 5.0,
    cmap_colors: List[ColorType] = ["r", "b"],
    label_points: bool = True,
    legend_loc: str = "best",
    **kwargs,
) -> Axes:
    r"""Return the ax with the points plotted based on x and y in a 1D plot. If
    pred_x/pred_y is not set, these are plotted as marks at the bottom of the plot,
    otherwise, each point is plotted as close as possible to the line defined by
    the prediction values (pred_x/pred_y). Usually use alongside `plot_predict_1d()`.

    Args:
        x (Iterable[Union[float, Iterable[float]]]): The `(n, 1)` or `(n, d, 2)` points to plot. The 3D case will
            be considered a pairwise plot.
        y (Iterable[float]): The `(n, 1)` responses to plot.
        ax (Axes, optional): The Matplotlib axes to plot onto. If not set, an axes is
            made and returned.
        pred_x (Iterable[float], optional): The points where the model was evaluated, used
            to position each point as close as possible to the line. If not set, the
            points are plotted as marks at the bottom of the plot.
        pred_y (Iterable[float], optional): The model outputs at each point in pred_x, used to
            position each point as close as possible to the line. If not set, the points
            are plotted as marks at the bottom of the plot.
        point_size (float): The size of each plotted point, defaults to 5.0.
        cmap_colors (List[ColorType]): A list of colors to map the point colors to from
            min to max of y. At least 2 colors are needed, but more colors will allow
            customizing intermediate colors. Defaults to ["r", "b"].
        label_points (bool): Add a way to identify the value of the points, whether as a
            legend for cases there are 6 or less unique responses or a colorbar for
            7 or more unique responses. Defaults to True.
        legend_loc (str): If a legend is added, where should it be placed.
        **kwargs: Extra kwargs passed to the ax.plot() call, note that every point is
            plotted with an individual call so these kwargs must be applicable to single
            points.

    Returns:
        Axes: The input axes with the points plotted onto it. Note that plotting is done
            in-place.
    """
    x, y, pred_x, pred_y = _tensor_cast(x, y, pred_x, pred_y)

    if len(cmap_colors) < 2:
        raise ValueError("cmap_colors must be at least 2 colors.")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cmap", cmap_colors)
    norm = matplotlib.colors.Normalize(y.min().item(), y.max().item())

    if ax is None:
        _, ax = plt.subplots()

    if len(x.shape) == 3:  # Multi dim case
        if pred_x is not None and pred_y is not None:
            for pair, response in zip(x, y):
                x1 = pair[:, 0]
                x2 = pair[:, 1]
                y1 = pred_y[torch.argmin(torch.abs(x1 - pred_x))]
                y2 = pred_y[torch.argmin(torch.abs(x2 - pred_x))]

                ax.plot(
                    [np.array(x1), np.array(x2)],
                    [np.array(y1), np.array(y2)],
                    "-",
                    c="gray",
                    alpha=0.5,
                )
                ax.plot(
                    x1,
                    y1,
                    marker="o",
                    color=cmap.reversed()(norm(response)),
                    markersize=point_size,
                    alpha=0.5,
                    **kwargs,
                )
                ax.plot(
                    x2,
                    y2,
                    marker="o",
                    color=cmap(norm(response)),
                    markersize=point_size,
                    alpha=0.5,
                    **kwargs,
                )

        else:

            def curve(start, end, mid):
                x_coords = (
                    start[0].item() if isinstance(start[0], torch.Tensor) else start[0],
                    end[0].item() if isinstance(end[0], torch.Tensor) else end[0],
                    mid[0].item() if isinstance(mid[0], torch.Tensor) else mid[0],
                )

                y_coords = (
                    start[1].item() if isinstance(start[1], torch.Tensor) else start[1],
                    end[1].item() if isinstance(end[1], torch.Tensor) else end[1],
                    mid[1].item() if isinstance(mid[1], torch.Tensor) else mid[1],
                )

                poly_x = np.array(x_coords).squeeze()
                poly_y = np.array(y_coords).squeeze()
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
                    color=cmap.reversed()(norm(response)),
                    **kwargs,
                )
                ax.plot(
                    x2,
                    hatch_y,
                    marker=3,
                    color=cmap(norm(response)),
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
                    color=cmap(norm(y_)),
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
                    color=cmap(norm(y_)),
                    **kwargs,
                )

    if label_points:
        _point_labeler(ax, y, cmap, norm, legend_loc)

    return ax


def plot_predict_2d(
    prediction: Iterable[Iterable[float]],
    lb: Iterable[float],
    ub: Iterable[float],
    ax: Optional[Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    edge_multiplier: float = 0.0,
    colorbar: bool = True,
    **kwargs,
) -> Axes:
    """Return the ax with the model predictions plotted in-place as a 2D heatmap.
    Usually used to plot the model outputs in posterior space or probability space.

    Args:
        prediction (Iterable[Iterable[float]]): A 2D array of the predictions, assumes it was a square
            parameter grid where each cell was evaluated by the model.
        lb (Iterable[float]): The lower bound of the two parameters being plotted.
        ub (Iterable[float]): The upper bound of the two parameters being plotted.
        ax (Axes, optional): The Matplotlib axes to plot onto. If not set, an axes is
            made and returned.
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
        colorbar (bool): Whether to add a colorbar (with bounds of [vmin, vmax]) to the
            parent figure of the Axes, defaults to True.
        **kwargs: Extra kwargs passed to the ax.imshow() call that creates the heatmap.

    Returns:
        Axes: The input axes with the predictions plotted onto it. Note that the plotting
            is done in-place.
    """
    prediction, lb, ub = _tensor_cast(prediction, lb, ub)
    if ax is None:
        _, ax = plt.subplots()

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

    return ax


def plot_points_2d(
    x: Iterable[Iterable[float]],
    y: Iterable[float],
    point_size: float = 5.0,
    ax: Optional[Axes] = None,
    axis: Optional[List[int]] = None,
    slice_vals: Optional[Iterable[float]] = None,
    slice_gradient: float = 1.0,
    cmap_colors: List[ColorType] = ["r", "b"],
    label_points: bool = True,
    legend_loc: str = "best",
    **kwargs,
) -> Axes:
    r"""Return the axes with the points defined by the parameters (x) and the outcomes
    (y) plotted in-place. If the inputs are from an experiment with stimuli per trial, a
    line is drawn between both. Usually used alongside `plot_predict_2d()`.

    Args:
        x (Iterable[Iterable[float]]): The `(n, d, 2)` or `(n, d)` data points to plot. Each point is
            a different trial.
        y (Iterable[float]): The `(n, 1)` responses given each set of datapoints. Each value
            is the response to a given trial.
        ax (Axes, optional): The Matplotlib axes to plot onto. If not set, an axes is
            made and returned.
        point_size (float): The size of the points, defaults to 5.
        axis (List[int], optional): If the dimensionality `d` is higher than 2, which
            two dimensions should the points be positioned with.
        slice_vals (Iterable[float]): If the dimensionality `d` is higher than 2, where was
            the other dimensions sliced at. This is used to determine the size/alpha
            gradient of the points based on how close each point is to the slice. If not
            set, the points will not change size/alpha based on slice distance. The
            Euclidean distance (s) of each point is calculated and converted to a
            multipler by `exp(-c * s)`, where c is the slice_gradient. The multipler is
            applied to the point_size and the alpha (transparency) of the points.
        slice_gradient (float): The rate at which the multiplier decreases as a function
            of the distance between a point and the slice. Defaults to 1.
        cmap_colors (List[ColorType]): A list of colors to map the point colors to from
            min to max of y. At least 2 colors are needed, but more colors will allow
            customizing intermediate colors. Defaults to ["r", "b"].
        label_points (bool): Add a way to identify the value of the points, whether as a
            legend for cases there are 6 or less unique responses or a colorbar for
            7 or more unique responses. Defaults to True.
        legend_loc (str): If a legend is added, where should it be placed.
        **kwargs: Extra kwargs passed to the ax.plot() call, note that every point is
            plotted with an individual call so these kwargs must be applicable to single
            points.

    Returns:
        Axes: The axes with the points plotted in-place.
    """
    x, y, slice_vals = _tensor_cast(x, y, slice_vals)

    if len(cmap_colors) < 2:
        raise ValueError("cmap_colors must be at least 2 colors.")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cmap", cmap_colors)
    norm = matplotlib.colors.Normalize(y.min().item(), y.max().item())

    if ax is None:
        _, ax = plt.subplots()

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
                color=cmap(norm(y_)),
                alpha=point_alpha[0].item(),
                markersize=point_alpha[0].item() * point_size,
                **kwargs,
            )
            ax.plot(
                xcoord[1],
                ycoord[1],
                marker="o",
                color=cmap.reversed()(norm(y_)),
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
                color=cmap(norm(y_)),
                alpha=point_alpha.item(),
                markersize=point_alpha.item() * point_size,
                **kwargs,
            )

    if label_points:
        _point_labeler(ax, y, cmap, norm, legend_loc)

    return ax


def plot_contours(
    prediction: Iterable[Iterable[float]],
    lb: Iterable[float],
    ub: Iterable[float],
    ax: Optional[Axes] = None,
    levels: Optional[Iterable[float]] = None,
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
        prediction (Iterable[Iterable[float]]): A 2D array of the predictions, assumes it was a square
            parameter grid where each cell was evaluated by the model.
        lb (Iterable[float]): The lower bound of the two parameters being plotted.
        ub (Iterable[float]): The upper bound of the two parameters being plotted.
        ax (Axes, optional): The Matplotlib axes to plot onto. If not set, an axes is
            made and returned.
        levels (Iterable[float], optional): A sequence of values to plot the contours given
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
    prediction, lb, ub, levels = _tensor_cast(prediction, lb, ub, levels)
    if ax is None:
        _, ax = plt.subplots()

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


def facet_slices(
    prediction: Union[torch.Tensor, np.ndarray],
    plotted_axes: List[int],
    lb: Iterable[float],
    ub: Iterable[float],
    nrows: int,
    ncols: int,
    plot_size: float,
    **kwargs,
) -> Tuple[matplotlib.figure.Figure, np.ndarray, np.ndarray, np.ndarray]:
    """Sets up a set of subplots to plot either a 3D or a 4D space where two dimensions
    are plotted and the other dimensions are sliced over the subplots.

    Args:
        prediction(Union[torch.Tensor, np.ndarray]): The model predictions cube to
            slice, in a 3D parameter space, it would be a 3D array, in a 4D parameter
            space it would be a 4D array.
        plotted_axes (List[int]): The two parameter indices that will be plotted, the
            other parameters will be sliced over subplots.
        lb (Iterable[Float]): The lower bound of the parameter space.
        ub (Iterable[Float]): The upper bound of the parameter space.
        nrows (int): How many rows to plot, which will also be how many slices there are
            of the first sliced dimension.
        ncols (int): How many columns to plot, which will also be how many slices there
            are of the second sliced dimesion.
        plot_size (float): The width of each individual square plot in inches.
        **kwargs: Kwargs passed to the plt.subplots() call.

    Returns:
        Figure: A Matplotlib figure of all of the subplots.
        np.ndarray[Axes]: 2D object array of each subplot.
        np.ndarray[torch.Tensor]: 2D object array of tensors representing the values of the sliced
            dimensions for each subplot.
        np.ndarray[torch.Tensor]: 2D object array of tensors representing the sliced predictions for
            each subplot.

    """
    prediction, lb, ub = _tensor_cast(prediction, lb, ub)
    not_axes = [i for i in range(len(lb)) if i not in plotted_axes]

    if len(not_axes) not in [1, 2]:
        raise ValueError("Only 3 and 4 dimensional spaces can use this function.")

    if "layout" in kwargs:
        layout = kwargs.pop("layout")
        warnings.warn(
            "The layout arg for subplots is defaulted to 'constrained', be careful when changing this"
        )
    else:
        layout = "constrained"

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(plot_size * ncols, plot_size * nrows),
        layout=layout,
        **kwargs,
    )

    slice_template = [
        None if idx in not_axes else slice(0, prediction.shape[idx])
        for idx in range(len(lb))
    ]
    slice_vals = np.empty_like(axes)
    slice_predictions = np.empty_like(axes)

    if len(axes.shape) > 1:
        row_slices = np.linspace(lb[not_axes[0]], ub[not_axes[0]], nrows)
        col_slices = np.linspace(lb[not_axes[1]], ub[not_axes[1]], ncols)
        row_idxs = np.linspace(0, prediction.shape[0] - 1, nrows, dtype=int)
        col_idxs = np.linspace(0, prediction.shape[0] - 1, ncols, dtype=int)

        for idx in np.ndindex(axes.shape):
            slice_vals[idx] = [row_slices[idx[0]], col_slices[idx[1]]]
            tmp_slice = slice_template[:]
            tmp_slice[tmp_slice.index(None)] = row_idxs[idx[0]]
            tmp_slice[tmp_slice.index(None)] = col_idxs[idx[1]]

            slice_predictions[idx] = prediction[tmp_slice]
    else:
        nPlots = nrows if nrows != 1 else ncols
        slices = np.linspace(lb[not_axes[0]], ub[not_axes[0]], nPlots)
        idxs = np.linspace(0, prediction.shape[0] - 1, nPlots, dtype=int)

        for idx in np.ndindex(axes.shape):
            slice_vals[idx] = slices[idx[0]]
            tmp_slice = slice_template[:]
            tmp_slice[tmp_slice.index(None)] = idxs[idx[0]]

            slice_predictions[idx] = prediction[tmp_slice]

    return fig, axes, slice_vals, slice_predictions


def _point_labeler(
    ax: Axes,
    responses: torch.Tensor,
    cmap: matplotlib.colors.Colormap,
    norm: matplotlib.colors.Normalize,
    legend_loc: str,
):
    # Given responses, create some way to indicate what point color means
    unique_responses = responses.unique()

    if len(unique_responses) <= 6:  # Make a legend, probably categorical
        handles = []
        for res in unique_responses:
            handle = matplotlib.lines.Line2D(
                [0],
                [0],
                label=np.around(res.item(), decimals=1),
                marker="o",
                linestyle="",
                color=cmap(norm(res)),
            )
            handles.append(handle)

        ax.legend(handles=handles, loc=legend_loc)
    else:  # Make a colorbar, probably continuous
        assert ax.figure is not None  # for mypy, unlikely to actually happen
        mappable = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        ax.figure.colorbar(mappable, ax=ax)


def _tensor_cast(*objs: Any) -> Tuple[torch.Tensor, ...]:
    # Turns objects into tensors if possible
    casted_objs: List[Any] = []
    for obj in objs:
        try:
            if not isinstance(obj, torch.Tensor) and hasattr(
                obj, "__iter__"
            ):  # Checks if iterable
                casted_objs.append(torch.tensor(obj))
            else:
                casted_objs.append(obj)
        except (ValueError, TypeError):
            casted_objs.append(obj)

    return tuple(casted_objs)
