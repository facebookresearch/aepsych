import unittest

import torch
from aepsych.config import Config
from aepsych.models.gp_classification import GPClassificationModel
from aepsych.models.inducing_point_allocators import (
    AutoAllocator,
    DummyAllocator,
    GreedyVarianceReduction,
    KMeansAllocator,
    SobolAllocator,
)
from aepsych.models.utils import select_inducing_points
from botorch.models.utils.inducing_point_allocators import GreedyImprovementReduction
from botorch.utils.sampling import draw_sobol_samples
from sklearn.datasets import make_classification


class TestInducingPointAllocators(unittest.TestCase):
    def test_sobol_allocator_from_config(self):
        config_str = """
            [common]
            parnames = [par1]

            [par1]
            par_type = continuous
            lower_bound = 0.0
            upper_bound = 1.0

        """
        config = Config()
        config.update(config_str=config_str)
        allocator = SobolAllocator.from_config(config)

        # Check if bounds are correctly loaded
        expected_bounds = torch.tensor([[0.0], [1.0]])
        self.assertTrue(torch.equal(allocator.bounds, expected_bounds))

    def test_kmeans_allocator_from_config(self):
        config_str = """
            [common]
            parnames = [par1]

            [par1]
            par_type = continuous
            lower_bound = 0.0
            upper_bound = 1.0

            [KMeansAllocator]
        """
        config = Config()
        config.update(config_str=config_str)
        allocator = KMeansAllocator.from_config(config)

        # No specific configuration to check, just test instantiation
        self.assertTrue(isinstance(allocator, KMeansAllocator))

    def test_auto_allocator_from_config_with_fallback(self):
        config_str = """
            [common]
            parnames = [par1]

            [par1]
            par_type = continuous
            lower_bound = 0.0
            upper_bound = 1.0

        """
        config = Config()
        config.update(config_str=config_str)
        allocator = AutoAllocator.from_config(config)

        # Check if fallback allocator is an instance of SobolAllocator with correct bounds
        self.assertTrue(isinstance(allocator.fallback_allocator, KMeansAllocator))

    def test_sobol_allocator_allocate_inducing_points(self):
        bounds = torch.tensor([[0.0], [1.0]])
        allocator = SobolAllocator(bounds=bounds)
        inducing_points = allocator.allocate_inducing_points(num_inducing=5)

        # Check shape and bounds of inducing points
        self.assertEqual(inducing_points.shape, (5, 1))
        self.assertTrue(
            torch.all(inducing_points >= bounds[0])
            and torch.all(inducing_points <= bounds[1])
        )

    def test_kmeans_allocator_allocate_inducing_points(self):
        inputs = torch.rand(100, 2)  # 100 points in 2D
        allocator = KMeansAllocator()
        inducing_points = allocator.allocate_inducing_points(
            inputs=inputs, num_inducing=10
        )

        # Check shape of inducing points
        self.assertEqual(inducing_points.shape, (10, 2))

    def test_auto_allocator_with_kmeans_fallback(self):
        inputs = torch.rand(50, 2)
        fallback_allocator = KMeansAllocator()
        allocator = AutoAllocator(fallback_allocator=fallback_allocator)
        inducing_points = allocator.allocate_inducing_points(
            inputs=inputs, num_inducing=10
        )

        # Check shape of inducing points and that fallback allocator is used
        self.assertEqual(inducing_points.shape, (10, 2))

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

    # Need to test instantiating a full model and checking if the inducing points are correct: initially dummy but with the correct inducing_point_method set. Then after fitting once, inducing points are replaced with real inducing points.
    def test_auto_allocator_allocate_inducing_points(self):
        train_X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        train_Y = torch.tensor([[1.0], [2.0], [3.0]])
        model = GPClassificationModel(
            lb=torch.tensor([0, 0]),
            ub=torch.tensor([4, 4]),
            inducing_point_method=AutoAllocator(),
            inducing_size=3,
        )
        self.assertTrue(model.last_inducing_points_method == "DummyAllocator")
        auto_inducing_points = AutoAllocator(
            bounds=torch.stack([torch.tensor([0, 0]), torch.tensor([4, 4])])
        ).allocate_inducing_points(
            inputs=train_X,
            covar_module=model.covar_module,
            num_inducing=model.inducing_size,
        )
        inital_inducing_points = DummyAllocator(
            bounds=torch.stack([torch.tensor([0, 0]), torch.tensor([4, 4])])
        ).allocate_inducing_points(
            inputs=train_X,
            covar_module=model.covar_module,
            num_inducing=model.inducing_size,
        )

        # Should be different from the initial inducing points
        self.assertFalse(
            torch.allclose(
                auto_inducing_points, model.variational_strategy.inducing_points
            )
        )
        self.assertTrue(
            torch.allclose(
                inital_inducing_points, model.variational_strategy.inducing_points
            )
        )

        model.fit(train_X, train_Y)
        self.assertTrue(model.last_inducing_points_method == "AutoAllocator")
        self.assertEqual(
            model.variational_strategy.inducing_points.shape, auto_inducing_points.shape
        )

        # Check that inducing points are updated after fitting
        self.assertTrue(
            torch.allclose(
                auto_inducing_points, model.variational_strategy.inducing_points
            )
        )


class TestGreedyAllocators(unittest.TestCase):
    def test_greedy_variance_reduction_allocate_inducing_points(self):
        # Mock data for testing
        train_X = torch.rand(100, 1)
        train_Y = torch.rand(100, 1)
        model = GPClassificationModel(
            lb=0,
            ub=1,
            inducing_point_method=GreedyVarianceReduction(),
            inducing_size=10,
        )

        # Instantiate GreedyVarianceReduction allocator
        allocator = GreedyVarianceReduction()

        # Allocate inducing points and verify output shape
        inducing_points = allocator.allocate_inducing_points(
            inputs=train_X,
            covar_module=model.covar_module,
            num_inducing=10,
            input_batch_shape=torch.Size([]),
        )
        inital_inducing_points = DummyAllocator(
            bounds=torch.stack([torch.tensor([0]), torch.tensor([1])])
        ).allocate_inducing_points(
            inputs=train_X,
            covar_module=model.covar_module,
            num_inducing=10,
            input_batch_shape=torch.Size([]),
        )
        self.assertEqual(inducing_points.shape, (10, 1))
        # Should be different from the initial inducing points
        self.assertFalse(
            torch.allclose(inducing_points, model.variational_strategy.inducing_points)
        )
        self.assertTrue(
            torch.allclose(
                inital_inducing_points, model.variational_strategy.inducing_points
            )
        )

        # Then fit the model and check that the inducing points are updated
        model.fit(train_X, train_Y)

        self.assertTrue(
            torch.allclose(inducing_points, model.variational_strategy.inducing_points)
        )


if __name__ == "__main__":
    unittest.main()
