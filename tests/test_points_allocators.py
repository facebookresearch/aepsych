import unittest

import torch
from aepsych.config import Config
from aepsych.models.gp_classification import GPClassificationModel
from aepsych.models.inducing_points import (
    AutoAllocator,
    FixedAllocator,
    GreedyVarianceReduction,
    KMeansAllocator,
    SobolAllocator,
)
from aepsych.models.utils import select_inducing_points
from aepsych.strategy import Strategy
from aepsych.transforms.parameters import ParameterTransforms, transform_options


class TestInducingPointAllocators(unittest.TestCase):
    def test_sobol_allocator_from_config(self):
        config_str = """
            [common]
            parnames = [par1]

            [par1]
            par_type = continuous
            lower_bound = 0.0
            upper_bound = 1.0
            log_scale = true

        """
        config = Config()
        config.update(config_str=config_str)
        allocator = SobolAllocator.from_config(config)

        # Check if bounds are correctly loaded
        expected_bounds = torch.tensor([[0.0], [1.0]])
        self.assertTrue(torch.equal(allocator.bounds, expected_bounds))

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
            acqf = MCLevelSetEstimation
            min_asks = 2
            model = GPClassificationModel

            [GPClassificationModel]
            inducing_point_method = SobolAllocator
            inducing_size = 2
        """

        config = Config()
        config.update(config_str=config_str)
        strat = Strategy.from_config(config, "init_strat")
        print(strat.model.inducing_point_method)
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

    def test_kmeans_allocator_from_config(self):
        config_str = """
            [common]
            parnames = [par1]

            [par1]
            par_type = continuous
            lower_bound = 0.0
            upper_bound = 1.0
            log_scale = true

            [KMeansAllocator]
        """
        config = Config()
        config.update(config_str=config_str)
        allocator = KMeansAllocator.from_config(config)

        self.assertTrue(isinstance(allocator, KMeansAllocator))
        self.assertTrue(allocator.dim == 1)

    def test_kmeans_allocator_allocate_inducing_points(self):
        inputs = torch.rand(100, 2)  # 100 points in 2D
        allocator = KMeansAllocator(dim=2)

        # Test dummy
        inducing_points = allocator.allocate_inducing_points(num_inducing=10)

        self.assertEqual(inducing_points.shape, (10, 2))
        self.assertIsNone(allocator.last_allocator_used)

        # Test real inducing points
        inducing_points = allocator.allocate_inducing_points(
            inputs=inputs, num_inducing=10
        )

        self.assertEqual(inducing_points.shape, (10, 2))
        self.assertIs(allocator.last_allocator_used, KMeansAllocator)

    def test_kmeans_allocator_from_model_config(self):
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
            acqf = MCLevelSetEstimation
            min_asks = 2
            model = GPClassificationModel

            [GPClassificationModel]
            inducing_point_method = KMeansAllocator
            inducing_size = 2
        """

        config = Config()
        config.update(config_str=config_str)
        strat = Strategy.from_config(config, "init_strat")
        self.assertTrue(isinstance(strat.model.inducing_point_method, KMeansAllocator))

    def test_auto_allocator_from_config(self):
        config_str = """
            [common]
            parnames = [par1]

            [par1]
            par_type = continuous
            lower_bound = 0.0
            upper_bound = 1.0
            log_scale = true

            [KMeansAllocator]
        """
        config = Config()
        config.update(config_str=config_str)
        allocator = AutoAllocator.from_config(config)

        self.assertTrue(isinstance(allocator, AutoAllocator))
        self.assertTrue(allocator.dim == 1)

    def test_auto_allocator_allocate_inducing_points(self):
        inputs = torch.rand(100, 2)  # 100 points in 2D
        allocator = AutoAllocator(dim=2)

        # Test dummy
        inducing_points = allocator.allocate_inducing_points(num_inducing=10)

        self.assertEqual(inducing_points.shape, (10, 2))
        self.assertIsNone(allocator.last_allocator_used)

        # Test real inducing points
        inducing_points = allocator.allocate_inducing_points(
            inputs=inputs, num_inducing=10
        )

        self.assertEqual(inducing_points.shape, (10, 2))
        self.assertIs(allocator.last_allocator_used, KMeansAllocator)

    def test_auto_allocator_from_model_config(self):
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
            acqf = MCLevelSetEstimation
            min_asks = 2
            model = GPClassificationModel

            [GPClassificationModel]
            inducing_point_method = AutoAllocator
            inducing_size = 2
        """

        config = Config()
        config.update(config_str=config_str)
        strat = Strategy.from_config(config, "init_strat")
        self.assertTrue(isinstance(strat.model.inducing_point_method, AutoAllocator))

    def test_greedy_variance_reduction_allocate_inducing_points(self):
        # Mock data for testing
        train_X = torch.rand(100, 1)
        train_Y = torch.rand(100, 1)
        lb = torch.tensor([0])
        ub = torch.tensor([1])
        bounds = torch.stack([lb, ub])
        model = GPClassificationModel(
            inducing_point_method=GreedyVarianceReduction(dim=1),
            inducing_size=10,
            dim=1,
        )

        # Instantiate GreedyVarianceReduction allocator
        allocator = GreedyVarianceReduction(dim=1)

        # Allocate inducing points and verify output shape
        inducing_points = allocator.allocate_inducing_points(
            inputs=train_X,
            covar_module=model.covar_module,
            num_inducing=10,
            input_batch_shape=torch.Size([]),
        )

        # Then fit the model and check that the inducing points are updated
        model.fit(train_X, train_Y)

        self.assertTrue(
            torch.allclose(inducing_points, model.variational_strategy.inducing_points)
        )

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
            acqf = MCLevelSetEstimation
            min_asks = 2
            model = GPClassificationModel

            [GPClassificationModel]
            inducing_point_method = GreedyVarianceReduction
            inducing_size = 2
        """

        config = Config()
        config.update(config_str=config_str)
        strat = Strategy.from_config(config, "init_strat")
        self.assertTrue(
            isinstance(strat.model.inducing_point_method, GreedyVarianceReduction)
        )

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
            acqf = MCLevelSetEstimation
            min_asks = 2
            model = GPClassificationModel

            [GPClassificationModel]
            inducing_point_method = FixedAllocator
            inducing_size = 2

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
            acqf = MCLevelSetEstimation
            min_asks = 2
            model = GPClassificationModel

            [GPClassificationModel]
            inducing_point_method = AutoAllocator
            inducing_size = 2
        """

        config = Config()
        config.update(config_str=config_str)
        strat = Strategy.from_config(config, "init_strat")
        self.assertIsInstance(strat.model.inducing_point_method, AutoAllocator)
        self.assertIsNone(strat.model.inducing_point_method.last_allocator_used)

        train_X = torch.tensor([[12.0, 0.0], [600.0, 1.0]])
        train_Y = torch.tensor([[1.0], [0.0]])

        # Fit the model and check that the inducing points are updated
        strat.add_data(train_X, train_Y)
        strat.fit()
        self.assertIs(
            strat.model.inducing_point_method.last_allocator_used, KMeansAllocator
        )
        self.assertEqual(
            strat.model.variational_strategy.inducing_points.shape, train_X.shape
        )

    def test_select_inducing_points_legacy(self):
        with self.assertWarns(DeprecationWarning):
            # Call select_inducing_points directly with a string for allocator to trigger the warning
            bounds = torch.tensor([[0.0], [1.0]])
            points = select_inducing_points(
                inducing_size=5,
                allocator="sobol",  # Legacy string argument to trigger DeprecationWarning
                bounds=bounds,
            )
            self.assertEqual(points.shape, (5, 1))


if __name__ == "__main__":
    unittest.main()
