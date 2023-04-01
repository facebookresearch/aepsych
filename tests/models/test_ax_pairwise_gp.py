from botorch.models import PairwiseLaplaceMarginalLogLikelihood
from aepsych.models import AxPairwiseGPModel
import unittest
import numpy as np
import numpy.testing as npt
import torch
from botorch.fit import fit_gpytorch_mll
from sklearn.datasets import make_classification


class AxPairwiseGPTestCase(unittest.TestCase):
    """
    Super basic smoke test to make sure we know if we broke the underlying model
    for pairwise model
    """

    def setUp(self):
        # np.random.seed(1)
        # torch.manual_seed(1)
        X, y = make_classification(
            n_samples=20,
            n_features=2,
            n_redundant=0,
            n_informative=1,
            random_state=1,
            n_clusters_per_class=1,
        )
        self.X, self.y = torch.Tensor(X), torch.Tensor(y).reshape(-1, 1)

        pairwise_model = AxPairwiseGPModel
        pairwise_model.dim = self.X.shape[1]
        pairwise_model.lb = torch.min(self.X, dim=0)[0]
        pairwise_model.ub = torch.max(self.X, dim=0)[0]
        self.pairwise_model = pairwise_model(None)

        self.datapoints, self.comparisons = self.pairwise_model._pairs_to_comparisons(
            self.X, self.y
        )
        self.pairwise_model.set_train_data(self.datapoints, self.comparisons)

    def test_None_rereference(self):
        """
        Just see if we memorize the training set with a None rereference
        """
        mll = PairwiseLaplaceMarginalLogLikelihood(
            self.pairwise_model.likelihood, self.pairwise_model
        )
        fit_gpytorch_mll(mll)

        # pspace
        with torch.no_grad():
            pm, pv = self.pairwise_model.predict(
                self.X, probability_space=True, rereference=None
            )
        pred = pm > 0.5
        npt.assert_allclose(pred.reshape(-1, 1), self.y)
        npt.assert_array_less(pv, 1)

        # fspace
        with torch.no_grad():
            pm, pv = self.pairwise_model.predict(
                self.X, probability_space=False, rereference=None
            )
        pred = pm > 0
        npt.assert_allclose(pred.reshape(-1, 1), self.y)

    def test_x_min_rereference(self):
        mll = PairwiseLaplaceMarginalLogLikelihood(
            self.pairwise_model.likelihood, self.pairwise_model
        )
        fit_gpytorch_mll(mll)

        # pspace
        with torch.no_grad():
            pm, pv = self.pairwise_model.predict(
                self.X, probability_space=True, rereference="x_min"
            )
        pred = pm > 0.5
        matches = np.isclose(pred.reshape(-1, 1), self.y)
        npt.assert_array_less(pv, 1)
        assert np.mean(matches) > 0.9, f"got {np.mean(matches)*100}% matches"

        # fspace
        with torch.no_grad():
            pm, pv = self.pairwise_model.predict(
                self.X, probability_space=False, rereference="x_min"
            )
        pred = pm > 0
        matches = np.isclose(pred.reshape(-1, 1), self.y)
        assert np.mean(matches) > 0.9, f"got {np.mean(matches)*100}% matches"

    def test_x_max_rereference(self):
        mll = PairwiseLaplaceMarginalLogLikelihood(
            self.pairwise_model.likelihood, self.pairwise_model
        )
        fit_gpytorch_mll(mll)

        # pspace
        with torch.no_grad():
            pm, pv = self.pairwise_model.predict(
                self.X, probability_space=True, rereference="x_max"
            )
        pred = pm < 0.5  # TODO: This only works when flipped is that correct?
        matches = np.isclose(pred.reshape(-1, 1), self.y)
        npt.assert_array_less(pv, 1)
        assert np.mean(matches) >= 0.75, f"got {np.mean(matches)*100}% matches"

        # fspace
        with torch.no_grad():
            pm, pv = self.pairwise_model.predict(
                self.X, probability_space=False, rereference="x_max"
            )
        pred = pm < 0
        matches = np.isclose(pred.reshape(-1, 1), self.y)
        assert np.mean(matches) >= 0.75, f"got {np.mean(matches)*100}% matches"


if __name__ == "__main__":
    unittest.main()
