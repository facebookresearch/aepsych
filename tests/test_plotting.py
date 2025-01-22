#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib

matplotlib.use("Agg")  # Ensures headless plotting

import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
from aepsych.models import GPClassificationModel, GPRegressionModel
from aepsych.plotting import (
    plot_contours,
    plot_points_1d,
    plot_points_2d,
    plot_predict_1d,
    plot_predict_2d,
)
from aepsych.transforms import ParameterTransformedModel, ParameterTransforms
from aepsych.transforms.ops import NormalizeScale
from aepsych.utils import dim_grid
from matplotlib.collections import PolyCollection
from sklearn.datasets import make_classification, make_regression


class TestClassificationPlotting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lb = torch.tensor([-10, 0, 100])
        cls.ub = torch.tensor([-5, 1, 900])
        dims = cls.lb.shape[0]
        cls.x, cls.y = make_classification(
            n_samples=100,
            n_features=dims,
            n_redundant=0,
            n_informative=3,
            random_state=1,
            n_clusters_per_class=4,
        )
        cls.x, cls.y = torch.tensor(cls.x), torch.tensor(cls.y)

        # Rescale x dimensions to specific bounds, first minmax scale normalize, then rescale with unnormalize with bounds
        _min_max = ParameterTransforms(normalize=NormalizeScale(d=dims))
        cls.x = _min_max.transform(cls.x)

        normalize = ParameterTransforms(
            normalize=NormalizeScale(d=3, bounds=torch.stack([cls.lb, cls.ub]))
        )
        cls.x = normalize.untransform(cls.x)

        # Make a model
        cls.model = ParameterTransformedModel(
            model=GPClassificationModel, dim=dims, transforms=normalize
        )

        # Add data to model
        cls.model.fit(cls.x, cls.y)

        # Get prediction grid
        cls.grid_size = 10
        cls.post_grid = dim_grid(cls.lb, cls.ub, gridsize=cls.grid_size)
        cls.posterior = cls.model.posterior(normalize.transform(cls.post_grid))
        cls.post_mean = cls.posterior.mean.squeeze().detach()

        cls.post_grid_1d = dim_grid(
            cls.lb, cls.ub, gridsize=cls.grid_size, slice_dims={1: 0.5, 2: 500}
        )
        cls.posterior_1d = cls.model.posterior(normalize.transform(cls.post_grid_1d))
        cls.post_mean_1d = cls.posterior_1d.mean.squeeze().detach()

        return cls

    def tearDown(self):
        plt.close()  # Saves memory

    def test_plot_predict_1d(self):
        samples = self.posterior_1d.sample(torch.Size([100])).squeeze()
        post_lower = torch.quantile(samples, 0.025, dim=0)
        post_upper = torch.quantile(samples, 0.975, dim=0)

        # First pass, check if args are respected
        fig, ax = plt.subplots()

        # Only plot 1D
        post_grid = self.post_grid_1d[:, 0]

        # Expected to plot in place, so original ax should be modified
        plotted_ax = plot_predict_1d(
            ax=ax,
            x=post_grid,
            prediction=self.post_mean_1d,
            pred_lower=post_lower,
            pred_upper=post_upper,
        )

        # Should only plot 1 line with the values from post_grid
        line = ax.get_lines()
        self.assertTrue(len(line) == 1)
        self.assertTrue(np.allclose(ax.get_lines()[0]._x, post_grid.numpy()))

        # Only one polygon should be drawn for error bars that max the extents of upper/lower
        poly = ax.findobj(PolyCollection)
        self.assertTrue(len(poly) == 1)
        Bbox = poly[0]._paths[0].get_extents()
        self.assertTrue(np.allclose(Bbox.x0, post_grid.min()))
        self.assertTrue(np.allclose(Bbox.x1, post_grid.max()))
        self.assertTrue(np.allclose(Bbox.y0, post_lower.min()))
        self.assertTrue(np.allclose(Bbox.y1, post_upper.max()))

        # Check plotted to original plot
        self.assertEqual(plotted_ax, fig.get_axes()[0])

        # Second pass smoketest for defaults working
        fig, ax = plt.subplots()

        plotted_ax = plot_predict_1d(x=post_grid, prediction=self.post_mean_1d)

        self.assertTrue(ax != plotted_ax)

        # Double checking that no error bar is drawn
        poly = ax.findobj(PolyCollection)
        self.assertTrue(len(poly) == 0)

    def test_plot_points_1d(self):
        fig, ax = plt.subplots()

        # Only plotting 1d case
        x = self.x[:, 0].numpy()
        post_grid = self.post_grid_1d[:, 0]

        point_0_color = "k"
        point_1_color = "g"
        cmap_colors = [point_0_color, point_1_color]
        point_size = 2.0

        # Plots dots on the posterior line
        plotted_ax = plot_points_1d(
            ax=ax,
            x=x,
            y=self.y,
            pred_x=post_grid,
            pred_y=self.post_mean_1d,
            point_size=point_size,
            cmap_colors=cmap_colors,
            label_points=False,
        )

        # Check we got the points right
        points = ax.get_lines()  # Each point is a separate line
        for i, point in enumerate(points):
            point_x, point_y = point.get_data()
            self.assertTrue(np.allclose(point_x, x[i]))
            self.assertTrue(
                np.allclose(
                    point_y,
                    self.post_mean_1d[torch.argmin(torch.abs(x[i] - post_grid))],
                )
            )
            self.assertEqual(point.get_markersize(), point_size)
            self.assertEqual(
                point.get_markerfacecolor(),
                (
                    matplotlib.colors.to_rgba(point_0_color)
                    if self.y[i] == 0
                    else matplotlib.colors.to_rgba(point_1_color)
                ),
            )
            self.assertEqual(point.get_marker(), "o")

        # There should be no legend
        self.assertIsNone(plotted_ax.get_legend())

        # Check plotted to original plot
        self.assertEqual(plotted_ax, fig.get_axes()[0])

        fig, ax = plt.subplots()

        # Plot hatch marks on the bottom using defaults for extra smoketest
        plotted_ax = plot_points_1d(
            ax=ax,
            x=x,
            y=self.y,
        )

        # Check we got the points right
        points = ax.get_lines()  # Each point is a separate line
        min_y, max_y = ax.get_ylim()
        for i, point in enumerate(points):
            point_x, point_y = point.get_data()
            self.assertTrue(np.allclose(point_x, x[i]))
            self.assertTrue(point_y > min_y and point_y < max_y)
            self.assertEqual(
                point.get_marker(), 3
            )  # Specifically marker 3, which is a line

        # Check plotted to original plot
        self.assertEqual(plotted_ax, fig.get_axes()[0])

        # Smoketest with defaults
        fig, ax = plt.subplots()

        plotted_ax = plot_points_1d(
            x=x,
            y=self.y,
        )

        # Should be legend with 2 entries
        self.assertTrue(len(plotted_ax.get_legend().legend_handles) == 2)

        self.assertTrue(plotted_ax != ax)

        with self.assertRaises(ValueError):
            _ = plot_points_1d(ax=ax, x=x, y=self.y, cmap_colors=["b"])

    def test_plot_points_1d_pairwise(self):
        _, ax = plt.subplots()

        # Simulate pair by stacking half
        x = torch.concatenate(
            [
                self.x[0 : self.x.shape[0] // 2, 0].unsqueeze(-1).unsqueeze(-1),
                self.x[self.x.shape[0] // 2 :, 0].unsqueeze(-1).unsqueeze(-1),
            ],
            axis=2,
        )
        y = self.y[: self.y.shape[0] // 2]
        post_grid = self.post_grid_1d[:, 0]

        point_0_color = "k"
        point_1_color = "g"
        point_size = 2.0

        # Plots dots on the posterior line, mostly a smoketest, main test checks for functionality
        _ = plot_points_1d(
            ax=ax,
            x=x,
            y=y,
            pred_x=post_grid,
            pred_y=self.post_mean_1d,
            point_size=point_size,
            cmap_colors=[point_0_color, point_1_color],
        )

        # Should be 150 line objects (100 points and 50 connecting lines)
        self.assertTrue(len(ax.get_lines()) == 150)

        _, ax = plt.subplots()

        # Plot hatch marks on the bottom using defaults for extra smoketest
        plotted_ax = plot_points_1d(
            x=x,
            y=y,
        )

        # Should be 150 line objects (100 points and 50 connecting lines)
        self.assertTrue(len(plotted_ax.get_lines()) == 150)

        self.assertTrue(plotted_ax != ax)

    def test_plot_predict_2d(self):
        fig, ax = plt.subplots()

        # Plotting 2D
        prediction = self.post_mean.reshape(*[self.grid_size] * len(self.lb))
        prediction = prediction[0]  # one slice
        lb = self.lb[1:]
        ub = self.ub[1:]
        vmin = -1
        vmax = 1
        edge_multiplier = 0.1
        colorbar = False

        plotted_ax = plot_predict_2d(
            ax=ax,
            prediction=prediction,
            lb=lb,
            ub=ub,
            vmin=vmin,
            vmax=vmax,
            edge_multiplier=edge_multiplier,
            colorbar=colorbar,
        )

        bumped_lb = lb.double()
        bumped_ub = ub.double()
        diff = np.abs(ub - lb)
        edge_bumps = (diff - (diff * (1 - edge_multiplier))) / 2
        bumped_lb -= edge_bumps
        bumped_ub += edge_bumps

        # Extents are flipped ordered in a very specific way to be natural
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        self.assertTrue(np.allclose(x0, bumped_lb[0]))
        self.assertTrue(np.allclose(x1, bumped_ub[0]))
        self.assertTrue(np.allclose(y0, bumped_lb[1]))
        self.assertTrue(np.allclose(y1, bumped_ub[1]))

        # Only one image should be added
        self.assertTrue(len(ax.get_images()) == 1)
        heatmap = ax.get_images()[0]

        cmin, cmax = heatmap.get_clim()
        self.assertTrue(cmin == vmin)
        self.assertTrue(cmax == vmax)
        self.assertTrue(heatmap.origin == "upper")
        self.assertTrue(
            np.allclose(  # Mimic what is done inside the function to natural extent
                heatmap.get_array().data, torch.flip(prediction.T, dims=[0]).numpy()
            )
        )

        # Check if no colobar was added
        self.assertTrue(len(fig.get_axes()) == 1)  # Only one axes should exist

        # Check plotted to original plot
        self.assertEqual(plotted_ax, fig.get_axes()[0])

        # Smoketests with defaults and no colorbar
        fig, ax = plt.subplots()

        plotted_ax = plot_predict_2d(
            prediction=prediction,
            lb=lb,
            ub=ub,
        )

        # Check if colobar was added
        self.assertTrue(
            len(plotted_ax.figure.get_axes()) == 2
        )  # An extra axes for colorbar

        self.assertTrue(plotted_ax != ax)

    def test_plot_points_2d(self):
        fig, ax = plt.subplots()

        with self.assertRaises(ValueError):  # Missing axis slice
            _ = plot_points_2d(ax=ax, x=self.x, y=self.y)

        point_size = 10
        axis = [1, 2]
        slice_vals = torch.tensor([-5])
        slice_gradient = 2.0
        point_0_color = "k"
        point_1_color = "g"

        plotted_ax = plot_points_2d(
            ax=ax,
            x=self.x,
            y=self.y,
            point_size=point_size,
            axis=axis,
            slice_vals=slice_vals,
            slice_gradient=slice_gradient,
            cmap_colors=[point_0_color, point_1_color],
            label_points=False,
        )

        # Each point is a line
        points = ax.get_lines()
        for i, point in enumerate(points):
            xy = np.hstack(point.get_data())
            self.assertTrue(np.allclose(xy, self.x[i, axis].numpy()))
            self.assertEqual(
                point.get_markerfacecolor(),
                (
                    matplotlib.colors.to_rgba(point_0_color)
                    if self.y[i] == 0
                    else matplotlib.colors.to_rgba(point_1_color)
                ),
            )

        # Check if alpha and size modifications actually happened
        self.assertTrue(any([point.get_markersize() < point_size for point in points]))
        self.assertTrue(any([[point.get_alpha() < 1 for point in points]]))

        self.assertIsNone(plotted_ax.get_legend())

        # Check plotted to original plot
        self.assertEqual(plotted_ax, fig.get_axes()[0])

        fig, ax = plt.subplots()

        # Smoketest with defaults
        plotted_ax = plot_points_2d(x=self.x, y=self.y, axis=axis)  # Unsliced default

        plotted_ax = plot_points_2d(
            x=self.x[:, 1:],  # Sliced default
            y=self.y,
        )

        # Check if defaults all hold
        points = plotted_ax.get_lines()
        self.assertTrue(all([point.get_markersize() == 5.0 for point in points]))
        self.assertTrue(all([[point.get_alpha() == 1 for point in points]]))

        # Legend with 2 entries
        self.assertTrue(len(plotted_ax.get_legend().legend_handles) == 2)

        self.assertTrue(plotted_ax != ax)

        with self.assertRaises(ValueError):
            _ = plot_points_2d(x=self.x, y=self.y, cmap_colors=["b"])

    def test_plot_points_2d_pairwise(self):
        _, ax = plt.subplots()

        # Simulate pair by stacking half
        x = torch.concatenate(
            [
                self.x[0 : self.x.shape[0] // 2, :].unsqueeze(-1),
                self.x[self.x.shape[0] // 2 :, :].unsqueeze(-1),
            ],
            axis=2,
        )
        y = self.y[: self.y.shape[0] // 2]

        point_size = 10
        axis = [1, 2]
        slice_vals = torch.tensor([-5])
        slice_gradient = 2.0
        point_0_color = "k"
        point_1_color = "g"

        # Other capabilities tested by main non-pairwise case, this is mostly smoketest
        plotted_ax = plot_points_2d(
            x=x,
            y=y,
            point_size=point_size,
            axis=axis,
            slice_vals=slice_vals,
            slice_gradient=slice_gradient,
            cmap_colors=[point_0_color, point_1_color],
        )

        # Should be 150 line objects (100 points and 50 connecting lines)
        self.assertTrue(len(plotted_ax.get_lines()) == 150)

        self.assertTrue(plotted_ax != ax)

    def test_plot_contours(self):
        fig, ax = plt.subplots()

        # Plotting 2D
        prediction = self.post_mean.reshape(*[self.grid_size] * len(self.lb))
        prediction = prediction[0]  # one slice
        lb = self.lb[1:]
        ub = self.ub[1:]
        levels = torch.tensor([-0.75, 0, 0.75])
        edge_multiplier = 0.1
        color = "black"
        labels = True
        linestyles = "dotted"

        plotted_ax = plot_contours(
            ax=ax,
            prediction=prediction,
            lb=lb,
            ub=ub,
            levels=levels,
            edge_multiplier=edge_multiplier,
            color=color,
            labels=labels,
            linestyles=linestyles,
        )

        # We expect 3 contours to be drawn
        contours = ax.collections[0]
        self.assertTrue(len(contours.get_paths()) == 3)

        color = np.vstack(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
        )  # Black
        self.assertTrue(np.allclose(contours.get_edgecolor(), color))

        self.assertTrue(  # Just looking for not solid lines
            all([linestyle[1] is not None for linestyle in contours.get_linestyles()])
        )

        # Check plotted to original plot
        self.assertEqual(plotted_ax, fig.get_axes()[0])

        fig, ax = plt.subplots()

        # Smoketest with defaults
        plotted_ax = plot_contours(
            prediction=prediction,
            lb=lb,
            ub=ub,
        )

        self.assertTrue(plotted_ax != fig)


class TestRegressionPlotting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dims = 3
        cls.lb = torch.zeros(cls.dims)
        cls.ub = torch.ones(cls.dims)
        cls.x, cls.y = make_regression(
            n_samples=100,
            n_features=cls.dims,
            n_informative=cls.dims,
            random_state=1,
        )
        cls.x, cls.y = torch.tensor(cls.x), torch.tensor(cls.y)

        transform = ParameterTransforms(normalize=NormalizeScale(d=cls.dims))

        # Make a model
        cls.model = ParameterTransformedModel(
            model=GPRegressionModel, dim=cls.dims, transforms=transform
        )

        # Add data to model
        cls.model.fit(cls.x, cls.y)

        # Need to freeze the transforms in place
        cls.model.eval()

        # Get prediction grid
        cls.grid_size = 10
        cls.post_grid = dim_grid(cls.lb, cls.ub, gridsize=cls.grid_size)
        cls.posterior = cls.model.posterior(cls.post_grid)
        cls.post_mean = cls.posterior.mean.squeeze().detach()

        cls.post_grid_1d = dim_grid(
            cls.lb, cls.ub, gridsize=cls.grid_size, slice_dims={1: 0.5, 2: 500}
        )
        cls.posterior_1d = cls.model.posterior(cls.post_grid_1d)
        cls.post_mean_1d = cls.posterior_1d.mean.squeeze().detach()

        return cls

    def tearDown(self):
        plt.close()  # Saves memory

    def test_plot_regression_points_1d(self):
        fig, ax = plt.subplots()

        # Only plotting 1d case
        x = self.x[:, 0].numpy()
        post_grid = self.post_grid_1d[:, 0]

        point_0_color = "r"
        point_1_color = "g"
        cmap_colors = [point_0_color, point_1_color]
        point_size = 10.0

        # Plots dots on the posterior line
        plotted_ax = plot_points_1d(
            ax=ax,
            x=x,
            y=self.y,
            pred_x=post_grid,
            pred_y=self.post_mean_1d,
            point_size=point_size,
            cmap_colors=cmap_colors,
            label_points=True,
        )

        # Check we got the points right
        points = ax.get_lines()  # Each point is a separate line
        for i, point in enumerate(points):
            point_x, point_y = point.get_data()
            self.assertTrue(np.allclose(point_x, x[i]))
            self.assertTrue(
                np.allclose(
                    point_y,
                    self.post_mean_1d[torch.argmin(torch.abs(x[i] - post_grid))],
                )
            )
            self.assertEqual(point.get_markersize(), point_size)
            self.assertEqual(point.get_marker(), "o")

        # There should be no legend
        self.assertIsNone(plotted_ax.get_legend())

        # Check if colobar was added
        self.assertTrue(
            len(plotted_ax.figure.get_axes()) == 2
        )  # An extra axes for colorbar

        # Check plotted to original plot
        self.assertEqual(plotted_ax, fig.get_axes()[0])

    def test_plot_regression_points_2d(self):
        fig, ax = plt.subplots()

        with self.assertRaises(ValueError):  # Missing axis slice
            _ = plot_points_2d(ax=ax, x=self.x, y=self.y)

        point_size = 10
        axis = [1, 2]
        slice_vals = torch.tensor([-5])
        slice_gradient = 2.0
        point_0_color = "r"
        point_1_color = "g"

        plotted_ax = plot_points_2d(
            ax=ax,
            x=self.x,
            y=self.y,
            point_size=point_size,
            axis=axis,
            slice_vals=slice_vals,
            slice_gradient=slice_gradient,
            cmap_colors=[point_0_color, point_1_color],
            label_points=True,
        )

        # Each point is a line
        points = ax.get_lines()
        for i, point in enumerate(points):
            xy = np.hstack(point.get_data())
            self.assertTrue(np.allclose(xy, self.x[i, axis].numpy()))

        # Check if alpha and size modifications actually happened
        self.assertTrue(any([point.get_markersize() < point_size for point in points]))
        self.assertTrue(any([[point.get_alpha() < 1 for point in points]]))

        self.assertIsNone(plotted_ax.get_legend())

        # Check if colobar was added
        self.assertTrue(
            len(plotted_ax.figure.get_axes()) == 2
        )  # An extra axes for colorbar

        # Check plotted to original plot
        self.assertEqual(plotted_ax, fig.get_axes()[0])
