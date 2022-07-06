import unittest

import numpy as np
import torch
from aepsych.config import Config
from aepsych.generators import PairwiseOptimizeAcqfGenerator, PairwiseSobolGenerator
from aepsych.models import PairwiseProbitModel
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption


class TestPairwiseOptimizeAcqfGenerator(unittest.TestCase):
    def test_instantiate_eubo(self):
        config = """
        [PairwiseOptimizeAcqfGenerator]
        acqf = AnalyticExpectedUtilityOfBestOption
        """
        generator = PairwiseOptimizeAcqfGenerator.from_config(Config(config_str=config))
        self.assertTrue(generator.acqf == AnalyticExpectedUtilityOfBestOption)

        # need a fitted model in order to instantiate the acqf successfully
        model = PairwiseProbitModel(lb=[-1], ub=[1])
        train_x = torch.Tensor([-0.5, 1, 0.5, -1]).reshape((2, 1, 2))
        train_y = torch.Tensor([0, 1])
        model.fit(train_x, train_y)
        acqf = generator._instantiate_acquisition_fn(model=model, train_x=train_x)
        self.assertTrue(isinstance(acqf, AnalyticExpectedUtilityOfBestOption))

    def test_pairwise_sobol_sizes(self):
        for dim in np.arange(1, 4):
            for nsamp in (3, 5, 7):
                generator = PairwiseSobolGenerator(
                    lb=np.arange(dim).tolist(), ub=(1 + np.arange(dim)).tolist()
                )
                shape_out = (nsamp, dim, 2)
                self.assertEqual(generator.gen(nsamp).shape, shape_out)


if __name__ == "__main__":
    unittest.main()
