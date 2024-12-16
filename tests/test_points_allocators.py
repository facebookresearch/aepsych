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
from aepsych.strategy import Strategy
from aepsych.transforms.parameters import ParameterTransforms, transform_options
from sklearn.datasets import make_classification


class TestInducingPointAllocators(unittest.TestCase):
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
        self.assertTrue(inducing_points.shape == (9, 2))
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

        self.assertTrue(strat.model.inducing_point_method.dim == 2)

    def test_auto_allocator_allocate_inducing_points(self):
        # Mock data for testing
        train_X = torch.randint(low=0, high=100, size=(100, 2), dtype=torch.float64)
        train_Y = torch.rand(100, 1)
        model = GPClassificationModel(
            inducing_point_method=AutoAllocator(dim=2),
            inducing_size=10,
            dim=2,
        )

        # Check if model has dummy points
        self.assertIsNone(model.inducing_point_method.last_allocator_used)
        self.assertTrue(torch.all(model.variational_strategy.inducing_points == 0))
        self.assertTrue(model.variational_strategy.inducing_points.shape == (10, 2))

        # Fit with small data leess than inducing_size
        model.fit(train_X[:9], train_Y[:9])

        # We still check for the base allocator
        self.assertIs(model.inducing_point_method.last_allocator_used, KMeansAllocator)
        inducing_points = model.variational_strategy.inducing_points
        self.assertTrue(inducing_points.shape == (9, 2))
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

    def test_auto_allocator_from_model_config(self):
        config_str = """
            [common]
            parnames = [par1, par2, par3]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat]

            [par1]
            par_type = continuous
            lower_bound = 10
            upper_bound = 1000
            log_scale = True

            [par2]
            par_type = binary

            [par3]
            par_type = fixed
            value = 10

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

        self.assertTrue(strat.model.inducing_point_method.dim == 2)

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

    def test_select_inducing_points(self):
        """Verify that when we have n_induc > data size, we use data as inducing,
        and otherwise we correctly select inducing points."""
        X, y = make_classification(
            n_samples=100,
            n_features=1,
            n_redundant=0,
            n_informative=1,
            random_state=1,
            n_clusters_per_class=1,
        )
        X, y = torch.Tensor(X), torch.Tensor(y)
        inducing_size = 20

        model = GPClassificationModel(
            dim=1,
            inducing_size=inducing_size,
            inducing_point_method=AutoAllocator(dim=1),
        )

        # Test dummy
        points = model.inducing_point_method.allocate_inducing_points(
            inputs=None,
            covar_module=model.covar_module,
            num_inducing=inducing_size,
        )
        self.assertTrue(torch.all(points == 0))

        model.set_train_data(X, y)

        points = model.inducing_point_method.allocate_inducing_points(
            inputs=model.train_inputs[0],
            covar_module=model.covar_module,
            num_inducing=inducing_size,
        )
        self.assertTrue(len(points) <= 20)

        allocator = GreedyVarianceReduction(dim=1)
        points = allocator.allocate_inducing_points(
            inputs=model.train_inputs[0],
            num_inducing=inducing_size,
            covar_module=model.covar_module,
        )
        self.assertTrue(len(points) <= 20)

        allocator = KMeansAllocator(dim=1)
        points = allocator.allocate_inducing_points(
            inputs=model.train_inputs[0],
            num_inducing=inducing_size,
            covar_module=model.covar_module,
        )
        self.assertEqual(len(points), 20)

        allocator = SobolAllocator(
            bounds=torch.stack([torch.tensor([0]), torch.tensor([1])]), dim=1
        )
        points = allocator.allocate_inducing_points(
            inputs=model.train_inputs[0],
            num_inducing=inducing_size,
            covar_module=model.covar_module,
        )
        self.assertTrue(len(points) <= 20)

        allocator = FixedAllocator(points=torch.tensor([[0], [1], [2], [3]]), dim=1)
        points = allocator.allocate_inducing_points(
            inputs=model.train_inputs[0],
            num_inducing=inducing_size,
            covar_module=model.covar_module,
        )
        self.assertTrue(len(points) <= 20)


if __name__ == "__main__":
    unittest.main()
