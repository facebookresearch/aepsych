import unittest

import torch
from aepsych.config import Config
from aepsych.models.inducing_point_allocators import (
    AutoAllocator,
    KMeansAllocator,
    SobolAllocator,
)
from aepsych.models.utils import select_inducing_points
from botorch.models import SingleTaskGP
from botorch.models.utils.inducing_point_allocators import (
    GreedyImprovementReduction,
    GreedyVarianceReduction,
    InducingPointAllocator,
)
from botorch.utils.sampling import draw_sobol_samples


class TestInducingPointAllocators(unittest.TestCase):
    def test_sobol_allocator_from_config(self):
        config_str = """
            [SobolAllocator]
            bounds = [[0.0, 1.0], [0.0, 1.0]]
        """
        config = Config()
        config.update(config_str=config_str)
        allocator = SobolAllocator.from_config(config)

        # Check if bounds are correctly loaded
        expected_bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        self.assertTrue(torch.equal(allocator.bounds, expected_bounds))

    def test_kmeans_allocator_from_config(self):
        config_str = """
            [KMeansAllocator]
        """
        config = Config()
        config.update(config_str=config_str)
        allocator = KMeansAllocator.from_config(config)

        # No specific configuration to check, just test instantiation
        self.assertTrue(isinstance(allocator, KMeansAllocator))

    def test_auto_allocator_from_config_with_fallback(self):
        config_str = """
            [AutoAllocator]
            fallback_allocator = SobolAllocator
            [SobolAllocator]
            bounds = [[0.0, 1.0], [0.0, 1.0]]
        """
        config = Config()
        config.update(config_str=config_str)
        allocator = AutoAllocator.from_config(config)

        # Check if fallback allocator is an instance of SobolAllocator with correct bounds
        self.assertTrue(isinstance(allocator.fallback_allocator, SobolAllocator))
        expected_bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        self.assertTrue(
            torch.equal(allocator.fallback_allocator.bounds, expected_bounds)
        )

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


class TestGreedyAllocators(unittest.TestCase):
    def test_greedy_variance_reduction_allocate_inducing_points(self):
        # Mock data for testing
        inputs = torch.rand(100, 2)
        model = SingleTaskGP(inputs, torch.sin(inputs.sum(dim=-1)).unsqueeze(-1))

        # Instantiate GreedyVarianceReduction allocator
        allocator = GreedyVarianceReduction()

        # Allocate inducing points and verify output shape
        inducing_points = allocator.allocate_inducing_points(
            inputs=inputs,
            covar_module=model.covar_module,
            num_inducing=10,
            input_batch_shape=torch.Size([]),
        )
        self.assertEqual(inducing_points.shape, (10, 2))

    def test_greedy_improvement_reduction_allocate_inducing_points(self):
        # Mock data for testing
        inputs = torch.rand(100, 2)
        model = SingleTaskGP(inputs, torch.sin(inputs.sum(dim=-1)).unsqueeze(-1))

        # Instantiate GreedyImprovementReduction allocator
        allocator = GreedyImprovementReduction(model=model, maximize=True)

        # Allocate inducing points and verify output shape
        inducing_points = allocator.allocate_inducing_points(
            inputs=inputs,
            covar_module=model.covar_module,
            num_inducing=10,
            input_batch_shape=torch.Size([]),
        )
        self.assertEqual(inducing_points.shape, (10, 2))


if __name__ == "__main__":
    unittest.main()
