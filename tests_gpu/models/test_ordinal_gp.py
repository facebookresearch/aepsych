import unittest

import numpy as np
import torch
from aepsych.generators import OptimizeAcqfGenerator, SobolGenerator
from aepsych.likelihoods import OrdinalLikelihood
from aepsych.models import OrdinalGPModel
from aepsych.strategy import SequentialStrategy, Strategy
from aepsych.transforms import (
    ParameterTransformedGenerator,
    ParameterTransformedModel,
    ParameterTransforms,
)
from aepsych.transforms.ops import NormalizeScale
from botorch.acquisition import qLogNoisyExpectedImprovement
from sklearn.datasets import make_regression


class OrdinalGPTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        torch.manual_seed(1)
        X, y, coeff = make_regression(
            n_samples=100,
            n_features=2,
            n_informative=2,
            random_state=1,
            coef=True,
        )

        self.X, self.y, self.coeff = (
            torch.Tensor(X),
            torch.Tensor(y),
            torch.Tensor(coeff),
        )

        # Remap y to an ordinal response via binning
        self.n_bins = 7
        _, self.edges = torch.histogram(self.y, bins=self.n_bins - 1)
        self.y = torch.bucketize(self.y, self.edges, right=True) - 1

        # Reused transform
        self.lb = torch.tensor([-3.0, -3.0])
        self.ub = torch.tensor([3.0, 3.0])
        self.transform = ParameterTransforms(
            normalize=NormalizeScale(d=2, bounds=torch.stack([self.lb, self.ub]))
        )

    def simulate_response(self, x, std=0.5):
        # Simulate noisy ordinal response
        x = x + torch.normal(torch.zeros_like(x), torch.ones_like(x) * std)
        y = torch.sum(x * self.coeff, dim=1)
        return torch.clamp(
            torch.bucketize(y, self.edges) - 1, min=0, max=self.n_bins - 1
        )

    def test_ordinal_strategy(self):
        n_init = 30
        n_opt = 2

        # Create strategy
        sobol_gen = ParameterTransformedGenerator(
            SobolGenerator,
            transforms=self.transform,
            lb=self.lb,
            ub=self.ub,
        )
        acqf_gen = ParameterTransformedGenerator(
            OptimizeAcqfGenerator,
            transforms=self.transform,
            lb=self.lb,
            ub=self.ub,
            acqf=qLogNoisyExpectedImprovement,
            acqf_kwargs={"prune_baseline": False},
        )
        model = ParameterTransformedModel(
            OrdinalGPModel,
            transforms=self.transform,
            dim=2,
            likelihood=OrdinalLikelihood(n_levels=self.n_bins),
        )

        strat_list = [
            Strategy(
                lb=self.lb,
                ub=self.ub,
                min_asks=n_init,
                generator=sobol_gen,
                stimuli_per_trial=1,
                outcome_types=["ordinal"],
                transforms=self.transform,
            ),
            Strategy(
                lb=self.lb,
                ub=self.ub,
                model=model,
                generator=acqf_gen,
                min_asks=n_opt,
                stimuli_per_trial=1,
                outcome_types=["ordinal"],
                transforms=self.transform,
                use_gpu_generating=True,
                use_gpu_modeling=True,
            ),
        ]
        strat = SequentialStrategy(strat_list)

        while not strat.finished:
            x = strat.gen(1)
            x = x.cpu()
            y = self.simulate_response(x)
            strat.add_data(x, y)

        model.cuda()
        # Test learned model
        min_, max_ = torch.argmax(
            strat.model.predict_probs(torch.stack([self.lb, self.ub])), axis=1
        )
        self.assertTrue(min_ == 0)
        self.assertTrue(max_ == 6)

        # Check model is on gpu
        self.assertEqual(strat.model.device.type, "cuda")
