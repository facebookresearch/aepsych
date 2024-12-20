#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from inspect import signature

import torch
from aepsych.acquisition import (
    ApproxGlobalSUR,
    EAVC,
    GlobalMI,
    GlobalSUR,
    LocalMI,
    LocalSUR,
    MCLevelSetEstimation,
    MCPosteriorVariance,
)
from aepsych.acquisition.lookahead import MOCU, SMOCU
from aepsych.acquisition.mutual_information import BernoulliMCMutualInformation
from aepsych.generators import OptimizeAcqfGenerator
from aepsych.models import GPClassificationModel
from aepsych.models.inducing_points import GreedyVarianceReduction
from aepsych.strategy import Strategy
from parameterized import parameterized

acqf_kwargs_target = {"target": 0.75}
acqf_kwargs_lookahead = {"target": 0.75, "lookahead_type": "posterior"}

acqfs = [
    (MCPosteriorVariance, {}),
    (ApproxGlobalSUR, acqf_kwargs_target),
    (MOCU, acqf_kwargs_target),
    (SMOCU, acqf_kwargs_target),
    (EAVC, acqf_kwargs_target),
    (EAVC, acqf_kwargs_lookahead),
    (GlobalMI, acqf_kwargs_target),
    (GlobalMI, acqf_kwargs_lookahead),
    (GlobalSUR, acqf_kwargs_target),
    (LocalMI, acqf_kwargs_target),
    (LocalSUR, acqf_kwargs_target),
    (MCLevelSetEstimation, acqf_kwargs_target),
    (BernoulliMCMutualInformation, {}),
]


class TestOptimizeAcqfGenerator(unittest.TestCase):
    @parameterized.expand(acqfs)
    def test_gpu_smoketest(self, acqf, acqf_kwargs):
        lb = torch.tensor([0.0])
        ub = torch.tensor([1.0])
        inducing_size = 10

        acqf_args_expected = list(signature(acqf).parameters.keys())
        if "lb" in acqf_args_expected:
            acqf_kwargs = acqf_kwargs.copy()
            acqf_kwargs["lb"] = lb
            acqf_kwargs["ub"] = ub

        model = GPClassificationModel(
            dim=1,
            inducing_size=inducing_size,
            inducing_point_method=GreedyVarianceReduction(dim=1),
        )

        generator = OptimizeAcqfGenerator(
            acqf=acqf,
            acqf_kwargs=acqf_kwargs,
            lb=torch.tensor([0.0]),
            ub=torch.tensor([1.0]),
        )

        strat = Strategy(
            lb=torch.tensor([0]),
            ub=torch.tensor([1]),
            model=model,
            generator=generator,
            stimuli_per_trial=1,
            outcome_types=["binary"],
            min_asks=1,
            use_gpu_modeling=True,
            use_gpu_generating=True,
        )

        strat.add_data(x=torch.tensor([0.90]), y=torch.tensor([1.0]))

        strat.gen(1)


if __name__ == "__main__":
    unittest.main()
