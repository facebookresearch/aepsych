# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
import os
from typing import List, Union

import numpy as np
import torch
from aepsych.benchmark.problem import LSEProblemWithEdgeLogging
from aepsych.benchmark.test_functions import (
    discrim_highdim,
    modified_hartmann6,
    novel_discrimination_testfun,
)
from aepsych.models import GPClassificationModel
from aepsych.models.inducing_points import KMeansAllocator

"""The DiscrimLowDim, DiscrimHighDim, ContrastSensitivity6d, and Hartmann6Binary classes
are copied from bernoulli_lse github repository (https://github.com/facebookresearch/bernoulli_lse)
by Letham et al. 2022."""


class DiscrimLowDim(LSEProblemWithEdgeLogging):
    name = "discrim_lowdim"
    bounds = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.double).T

    def __init__(
        self, thresholds: Union[float, List, torch.Tensor, None] = None
    ) -> None:
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

    def __init__(
        self, thresholds: Union[float, List, torch.Tensor, None] = None
    ) -> None:
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

    def __init__(
        self, thresholds: Union[float, List, torch.Tensor, None] = None
    ) -> None:
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

    def __init__(
        self, thresholds: Union[float, List, torch.Tensor, None] = None
    ) -> None:
        thresholds = 0.75 if thresholds is None else thresholds
        super().__init__(thresholds=thresholds)

        # Load the data
        self.data = np.loadtxt(
            os.path.join("..", "..", "dataset", "csf_dataset.csv"),
            delimiter=",",
            skiprows=1,
        )
        y = torch.LongTensor(self.data[:, 0])
        x = torch.Tensor(self.data[:, 1:])
        inducing_size = 100

        # Fit a model, with a large number of inducing points
        self.m = GPClassificationModel(
            dim=6,
            inducing_size=inducing_size,
            inducing_point_method=KMeansAllocator(dim=6),
        )

        self.m.fit(
            x,
            y,
        )

    def f(self, X: torch.Tensor) -> torch.Tensor:
        # clamp f to 0 since we expect p(x) to be lower-bounded at 0.5
        return torch.clamp(self.m.predict(torch.tensor(X))[0], min=0)
