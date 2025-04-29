#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from aepsych.config import Config
from aepsych.models.gp_classification import GPClassificationModel
from aepsych.models.inducing_points import (
    DataAllocator,
    FixedAllocator,
    FixedPlusAllocator,
    GreedyVarianceReduction,
    KMeansAllocator,
    SobolAllocator,
)
from aepsych.strategy import SequentialStrategy, Strategy
from aepsych.transforms.parameters import ParameterTransforms, transform_options
from sklearn.datasets import make_classification


class TestBaseAllocator(unittest.TestCase):
    """Tests for the base allocator functionality and common behaviors."""

    def test_model_default_allocator(self):
        model = GPClassificationModel(dim=2)

        self.assertIsInstance(model.inducing_point_method, GreedyVarianceReduction)
        self.assertTrue(model.inducing_point_method.dim == 2)

    def test_allocator_model_fit(self):
        config_str = """
            [common]
            parnames = [par1, par2]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat]

            [par1]
            par_type = continuous
            lower_bound = 10
            upper_bound = 1000
            log_scale = True

            [par2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = OptimizeAcqfGenerator
            min_asks = 2
            model = GPClassificationModel

            [OptimizeAcqfGenerator]
            acqf = MCLevelSetEstimation

            [GPClassificationModel]
            inducing_size = 2
        """

        config = Config()
        config.update(config_str=config_str)
        strat = Strategy.from_config(config, "init_strat")
        self.assertIsInstance(
            strat.model.inducing_point_method, GreedyVarianceReduction
        )
        self.assertIsNone(strat.model.inducing_point_method.last_allocator_used)

        train_X = torch.tensor([[12.0, 0.0], [600.0, 1.0]])
        train_Y = torch.tensor([[1.0], [0.0]])

        # Fit the model and check that the inducing points are updated
        strat.add_data(train_X, train_Y)
        strat.fit()
        self.assertIs(
            strat.model.inducing_point_method.last_allocator_used,
            GreedyVarianceReduction,
        )
        self.assertEqual(
            strat.model.variational_strategy.inducing_points.shape, train_X.shape
        )

    def test_dummy_points(self):
        """Test that allocators return dummy points when inputs is None."""
        inducing_size = 20
        model = GPClassificationModel(
            dim=1,
            inducing_size=inducing_size,
            inducing_point_method=GreedyVarianceReduction(dim=1),
        )

        # Find dummy points
        points = model.variational_strategy.inducing_points

        self.assertTrue(torch.all(points == 0))
        self.assertIsNone(model.inducing_point_method.last_allocator_used)


class TestSobolAllocator(unittest.TestCase):
    """Tests for the SobolAllocator class."""

    def test_sobol_allocator_allocate_inducing_points(self):
        bounds = torch.tensor([[0.0], [1.0]])
        allocator = SobolAllocator(bounds=bounds, dim=1)
        inducing_points = allocator.allocate_inducing_points(num_inducing=5)

        # Check shape and bounds of inducing points
        self.assertEqual(inducing_points.shape, (5, 1))
        self.assertTrue(
            torch.all(inducing_points >= bounds[0])
            and torch.all(inducing_points <= bounds[1])
        )
        self.assertIs(allocator.last_allocator_used, SobolAllocator)

    def test_sobol_allocator_from_model_config(self):
        config_str = """
            [common]
            parnames = [par1]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat]

            [par1]
            par_type = continuous
            lower_bound = 10
            upper_bound = 1000
            log_scale = True

            [init_strat]
            generator = OptimizeAcqfGenerator
            min_asks = 2
            model = GPClassificationModel

            [GPClassificationModel]
            inducing_point_method = SobolAllocator
            inducing_size = 2

            [OptimizeAcqfGenerator]
            acqf = MCLevelSetEstimation
        """

        config = Config()
        config.update(config_str=config_str)
        strat = Strategy.from_config(config, "init_strat")
        self.assertTrue(isinstance(strat.model.inducing_point_method, SobolAllocator))

        # check that the bounds are scaled correctly
        self.assertFalse(
            torch.equal(
                strat.model.inducing_point_method.bounds, torch.tensor([[10], [1000]])
            )
        )
        self.assertTrue(
            torch.equal(
                strat.model.inducing_point_method.bounds, torch.tensor([[0], [1]])
            )
        )


class TestKMeansAllocator(unittest.TestCase):
    """Tests for the KMeansAllocator class."""

    def test_kmeans_allocator_allocate_inducing_points(self):
        # Mock data for testing
        train_X = torch.randint(low=0, high=100, size=(100, 2), dtype=torch.float64)
        train_Y = torch.rand(100, 1)
        model = GPClassificationModel(
            inducing_point_method=KMeansAllocator(dim=2),
            inducing_size=10,
            dim=2,
        )

        # Check if model has dummy points
        self.assertIsNone(model.inducing_point_method.last_allocator_used)
        self.assertTrue(torch.all(model.variational_strategy.inducing_points == 0))
        self.assertTrue(model.variational_strategy.inducing_points.shape == (10, 2))

        # Fit with small data leess than inducing_size
        model.fit(train_X[:9], train_Y[:9])

        self.assertIs(model.inducing_point_method.last_allocator_used, KMeansAllocator)
        inducing_points = model.variational_strategy.inducing_points
        self.assertTrue(
            inducing_points.shape == (9, 2), f"shape is {inducing_points.shape}"
        )
        # We made ints, so mod 1 should be 0s, so we know these were the original inputs
        self.assertTrue(torch.all(inducing_points % 1 == 0))

        # Then fit the model and check that the inducing points are updated
        model.fit(train_X, train_Y)

        self.assertIs(model.inducing_point_method.last_allocator_used, KMeansAllocator)
        inducing_points = model.variational_strategy.inducing_points
        self.assertTrue(inducing_points.shape == (10, 2))
        # It's highly unlikely clustered will all be integers, so check against extents too
        self.assertFalse(torch.all(inducing_points % 1 == 0))
        self.assertTrue(torch.all((inducing_points >= 0) & (inducing_points <= 100)))

    def test_kmeans_allocator_from_model_config(self):
        config_str = """
            [common]
            parnames = [par1, par2]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat]

            [par1]
            par_type = continuous
            lower_bound = 10
            upper_bound = 1000
            log_scale = True

            [par2]
            par_type = integer
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = OptimizeAcqfGenerator
            min_asks = 2
            model = GPClassificationModel

            [GPClassificationModel]
            inducing_point_method = KMeansAllocator
            inducing_size = 2

            [OptimizeAcqfGenerator]
            acqf = MCLevelSetEstimation
        """

        config = Config()
        config.update(config_str=config_str)
        strat = Strategy.from_config(config, "init_strat")
        self.assertTrue(isinstance(strat.model.inducing_point_method, KMeansAllocator))

        self.assertTrue(strat.model.inducing_point_method.dim == 2)

    def test_kmeans_shape_handling(self):
        allocator = KMeansAllocator(dim=1)

        inputs = torch.tensor([[1], [2], [3]])

        inputs_aug = torch.hstack([inputs, torch.zeros(size=[3, 1])])

        points = allocator.allocate_inducing_points(inputs=inputs_aug, num_inducing=2)
        self.assertTrue(points.shape == (2, 1))

        points = allocator.allocate_inducing_points(inputs=inputs_aug, num_inducing=100)
        self.assertTrue(torch.equal(points, inputs))


class TestGreedyVarianceReduction(unittest.TestCase):
    """Tests for the GreedyVarianceReduction class."""

    def test_greedy_variance_allocator_no_covar_raise(self):
        allocator = GreedyVarianceReduction(dim=2)

        with self.assertRaises(ValueError):
            _ = allocator.allocate_inducing_points(
                inputs=torch.zeros((30, 1)), num_inducing=10
            )

    def test_greedy_variance_reduction_allocate_inducing_points(self):
        # Mock data for testing
        train_X = torch.randint(low=0, high=100, size=(100, 2), dtype=torch.float64)
        train_Y = torch.rand(100, 1)
        model = GPClassificationModel(
            inducing_point_method=GreedyVarianceReduction(dim=2),
            inducing_size=10,
            dim=2,
        )

        # Check if model has dummy points
        self.assertIsNone(model.inducing_point_method.last_allocator_used)
        self.assertTrue(torch.all(model.variational_strategy.inducing_points == 0))
        self.assertTrue(model.variational_strategy.inducing_points.shape == (10, 2))

        # Then fit the model and check that the inducing points are updated
        model.fit(train_X, train_Y)

        self.assertIs(
            model.inducing_point_method.last_allocator_used, GreedyVarianceReduction
        )
        inducing_points = model.variational_strategy.inducing_points
        self.assertTrue(inducing_points.shape == (10, 2))
        self.assertTrue(torch.all((inducing_points >= 0) & (inducing_points <= 100)))

    def test_greedy_variance_from_config(self):
        config_str = """
            [common]
            parnames = [par1]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat]

            [par1]
            par_type = continuous
            lower_bound = 10
            upper_bound = 1000
            log_scale = True

            [init_strat]
            generator = OptimizeAcqfGenerator
            min_asks = 2
            model = GPClassificationModel

            [GPClassificationModel]
            inducing_point_method = GreedyVarianceReduction
            inducing_size = 2

            [OptimizeAcqfGenerator]
            acqf = MCLevelSetEstimation
        """

        config = Config()
        config.update(config_str=config_str)
        strat = Strategy.from_config(config, "init_strat")
        self.assertTrue(
            isinstance(strat.model.inducing_point_method, GreedyVarianceReduction)
        )


class TestFixedAllocator(unittest.TestCase):
    """Tests for the FixedAllocator class."""

    def test_fixed_allocator_allocate_inducing_points(self):
        config_str = """
            [common]
            parnames = [par1]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat]

            [par1]
            par_type = continuous
            lower_bound = 10
            upper_bound = 1000
            log_scale = True

            [init_strat]
            generator = OptimizeAcqfGenerator
            min_asks = 2
            model = GPClassificationModel

            [GPClassificationModel]
            inducing_point_method = FixedAllocator
            inducing_size = 2

            [OptimizeAcqfGenerator]
            acqf = MCLevelSetEstimation

            [FixedAllocator]
            points = [[0.1], [0.2]]
        """

        config = Config()
        config.update(config_str=config_str)
        strat = Strategy.from_config(config, "init_strat")
        self.assertTrue(isinstance(strat.model.inducing_point_method, FixedAllocator))

        # Check that the inducing points are the same as the fixed points (pre-transformation)
        inducing_points_pre_transform = FixedAllocator(
            points=torch.tensor([[0.1], [0.2]]), dim=1
        ).allocate_inducing_points(num_inducing=2)
        self.assertTrue(
            torch.equal(inducing_points_pre_transform, torch.tensor([[0.1], [0.2]]))
        )

        # Check that the inducing points are not the same as the fixed points (post-transformation)
        inducing_points_after_transform = (
            strat.model.inducing_point_method.allocate_inducing_points(num_inducing=2)
        )
        self.assertFalse(
            torch.equal(inducing_points_after_transform, torch.tensor([[0.1], [0.2]]))
        )

        # make the transformation
        transforms = ParameterTransforms.from_config(config)
        transformed_config = transform_options(config, transforms)
        transformed_points = torch.tensor(
            eval(transformed_config["FixedAllocator"]["points"])
        )
        # Check that the inducing points are the same as the fixed points (post-transformation)
        self.assertTrue(
            torch.equal(inducing_points_after_transform, transformed_points)
        )

        # Fit the model and check that the inducing points are updated
        train_X = torch.tensor([[6.0], [3.0]])
        train_Y = torch.tensor([[1.0], [1.0]])
        strat.add_data(train_X, train_Y)
        strat.fit()
        self.assertTrue(
            torch.equal(
                strat.model.variational_strategy.inducing_points, transformed_points
            )
        )


class TestFixedPlusAllocator(unittest.TestCase):
    """Tests for the FixedPlusAllocator class."""

    def test_fixed_plus_allocator(self):
        fixed_points = torch.tensor([[-0.1, -0.1], [1.1, 1.1]])

        allocator = FixedPlusAllocator(
            dim=2,
            points=fixed_points,
            main_allocator=KMeansAllocator,
        )

        x = torch.rand((10, 2))

        points = allocator.allocate_inducing_points(inputs=x, num_inducing=10)

        # Shoud be 12 points with fixed in there somewhere
        self.assertEqual(points.shape[0], 12)
        for point in fixed_points:
            # Check if point is a row in points
            self.assertTrue(
                torch.any(torch.vmap(lambda x, p=point: torch.all(x == p))(points))
            )

        self.assertTrue(torch.all(torch.isin(fixed_points, points)))

        x = torch.concat([torch.rand((10, 2)), fixed_points], dim=0)
        points = allocator.allocate_inducing_points(inputs=x, num_inducing=12)

        # Should still be 12 points with fixed inside
        self.assertEqual(points.shape[0], 12)
        for point in fixed_points:
            # Check if point is a row in points
            self.assertTrue(
                torch.any(torch.vmap(lambda x, p=point: torch.all(x == p))(points))
            )

    def test_fixed_plus_allocator_dimension_mismatch(self):
        fixed_points = torch.tensor([[-0.1, -0.1, -0.1], [1.1, 1.1, 1.1]])

        with self.assertRaises(ValueError):
            FixedPlusAllocator(
                dim=2,
                points=fixed_points,
                main_allocator=KMeansAllocator,
            )


class TestDataAllocator(unittest.TestCase):
    """Tests for the DataAllocator class."""

    def test_data_allocator(self):
        """Test basic functionality of DataAllocator."""
        allocator = DataAllocator(dim=2)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Test that it returns the input data and sets last_allocator_used
        inducing_points = allocator.allocate_inducing_points(
            inputs=inputs, num_inducing=10
        )
        self.assertTrue(torch.equal(inducing_points, inputs))
        self.assertIs(allocator.last_allocator_used, DataAllocator)
        self.assertIsNot(inducing_points, inputs)  # Should be a clone

        # Test when no inputs are provided we get dummy points
        inducing_points = allocator.allocate_inducing_points(num_inducing=10)
        self.assertEqual(inducing_points.shape, (10, 2))
        self.assertTrue(torch.all(inducing_points == 0))

        # Test warning when num_inducing is less than inputs
        with self.assertWarns(UserWarning) as w:
            inducing_points = allocator.allocate_inducing_points(
                inputs=inputs, num_inducing=2
            )

        self.assertEqual(len(w.warnings), 1)
        self.assertIn("DataAllocator ignores num_inducing=2", w.warning.args[0])
        self.assertTrue(torch.all(inducing_points == inputs))

    def test_data_allocator_config_smoketest(self):
        """Test DataAllocator integration with model and config."""
        # Test with config
        config_str = """
            [common]
            parnames = [par1]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat, opt_strat]

            [par1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = SobolGenerator
            min_asks = 2

            [opt_strat]
            generator = OptimizeAcqfGenerator
            min_asks = 1
            model = GPClassificationModel

            [GPClassificationModel]
            inducing_point_method = DataAllocator
            inducing_size = 2

            [OptimizeAcqfGenerator]
            acqf = MCLevelSetEstimation
        """

        config = Config()
        config.update(config_str=config_str)
        strat = SequentialStrategy.from_config(config)

        for response in [0, 1]:
            point = strat.gen()
            strat.add_data(point, torch.tensor([response]))

        point = strat.gen()
        self.assertTrue(
            torch.all(strat.model.variational_strategy.inducing_points == strat.x)
        )


if __name__ == "__main__":
    unittest.main()
