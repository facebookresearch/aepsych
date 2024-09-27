# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
import os

import numpy as np
import torch
from aepsych.benchmark.problem import LSEProblemWithEdgeLogging
from aepsych.benchmark.test_functions import (
    discrim_highdim,
    modified_hartmann6,
    novel_discrimination_testfun,
)
from aepsych.models import GPClassificationModel
from scipy.stats import norm

"""The DiscrimLowDim, DiscrimHighDim, ContrastSensitivity6d, and Hartmann6Binary classes
are copied from bernoulli_lse github repository (https://github.com/facebookresearch/bernoulli_lse)
by Letham et al. 2022."""


class DiscrimLowDim(LSEProblemWithEdgeLogging):
    name = "discrim_lowdim"
    bounds = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.double).T

    def __init__(self, thresholds=None):
        thresholds = 0.75 if thresholds is None else thresholds
        super().__init__(thresholds=thresholds)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return novel_discrimination_testfun(x).to(torch.double)  # type: ignore


class DiscrimHighDim(LSEProblemWithEdgeLogging):
    name = "discrim_highdim"
    bounds = torch.tensor(
        [
            [-1, 1],
            [-1, 1],
            [0.5, 1.5],
            [0.05, 0.15],
            [0.05, 0.2],
            [0, 0.9],
            [0, 3.14 / 2],
            [0.5, 2],
        ],
        dtype=torch.double,
    ).T

    def __init__(self, thresholds=None):
        thresholds = 0.75 if thresholds is None else thresholds
        super().__init__(thresholds=thresholds)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(discrim_highdim(x), dtype=torch.double)


class Hartmann6Binary(LSEProblemWithEdgeLogging):
    name = "hartmann6_binary"
    bounds = torch.stack(
        (
            torch.zeros(6, dtype=torch.double),
            torch.ones(6, dtype=torch.double),
        )
    )

    def __init__(self, thresholds=None):
        thresholds = 0.5 if thresholds is None else thresholds
        super().__init__(thresholds=thresholds)

    def f(self, X: torch.Tensor) -> torch.Tensor:
        y = torch.tensor([modified_hartmann6(x) for x in X], dtype=torch.double)
        f = 3 * y - 2.0
        return f


class ContrastSensitivity6d(LSEProblemWithEdgeLogging):
    """
    Uses a surrogate model fit to real data from a constrast sensitivity study.
    """

    name = "contrast_sensitivity_6d"
    bounds = torch.tensor(
        [[-1.5, 0], [-1.5, 0], [0, 20], [0.5, 7], [1, 10], [0, 10]],
        dtype=torch.double,
    ).T

    def __init__(self, thresholds=None):
        thresholds = 0.75 if thresholds is None else thresholds

        # Load the data
        self.data = np.loadtxt(
            os.path.join("..", "..", "dataset", "csf_dataset.csv"),
            delimiter=",",
            skiprows=1,
        )
        y = torch.LongTensor(self.data[:, 0])
        x = torch.Tensor(self.data[:, 1:])

        # Fit a model, with a large number of inducing points
        self.m = GPClassificationModel(
            lb=self.bounds[0],
            ub=self.bounds[1],
            inducing_size=100,
            inducing_point_method="kmeans++",
        )

        self.m.fit(
            x,
            y,
        )

        super().__init__(thresholds=thresholds)

    def f(self, X: torch.Tensor) -> torch.Tensor:
        # clamp f to 0 since we expect p(x) to be lower-bounded at 0.5
        return torch.clamp(self.m.predict(torch.tensor(X))[0], min=0)


class PairwiseDiscrimLowdim(LSEProblemWithEdgeLogging):
    name = "pairwise_discrim_lowdim"
    bounds = torch.tensor([[-1, 1], [-1, 1], [-1, 1], [-1, 1]], dtype=torch.double).T

    def __init__(self, thresholds=None):
        if thresholds is None:
            jnds = np.arange(-4, 5)
            thresholds = np.round(norm.cdf(jnds).tolist(), 3).tolist()
        super().__init__(thresholds=thresholds)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 4
        f1 = novel_discrimination_testfun(x[..., :2])
        f2 = novel_discrimination_testfun(x[..., 2:])
        return (f1 - f2).to(torch.double)


class PairwiseDiscrimHighdim(LSEProblemWithEdgeLogging):
    name = "pairwise_discrim_highdim"
    bounds = torch.tensor(
        [
            [-1, 1],
            [-1, 1],
            [0.5, 1.5],
            [0.05, 0.15],
            [0.05, 0.2],
            [0, 0.9],
            [0, 3.14 / 2],
            [0.5, 2],
        ]
        * 2,
        dtype=torch.double,
    ).T

    def __init__(self, thresholds=None):
        if thresholds is None:
            jnds = np.arange(-4, 5)
            thresholds = np.round(norm.cdf(jnds).tolist(), 3).tolist()
        super().__init__(thresholds=thresholds)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 16
        f1 = discrim_highdim(x[..., :8])
        f2 = discrim_highdim(x[..., 8:])
        return torch.tensor(f1 - f2, dtype=torch.double)


class PairwiseHartmann6Binary(LSEProblemWithEdgeLogging):
    name = "pairwise_hartmann6_binary"
    bounds = torch.stack(
        (
            torch.zeros(12, dtype=torch.double),
            torch.ones(12, dtype=torch.double),
        )
    )

    def __init__(self, thresholds=None):
        if thresholds is None:
            jnds = np.arange(-4, 5)
            thresholds = np.round(norm.cdf(jnds).tolist(), 3).tolist()
        super().__init__(thresholds=thresholds)

    def f(self, X: torch.Tensor) -> torch.Tensor:
        assert X.shape[-1] == 12

        def latent_f(X1):
            y = torch.tensor([modified_hartmann6(x) for x in X1], dtype=torch.double)
            f = 3 * y - 2.0
            return f

        f1 = latent_f(X[..., :6])
        f2 = latent_f(X[..., 6:])
        return (f1 - f2).to(torch.double)
