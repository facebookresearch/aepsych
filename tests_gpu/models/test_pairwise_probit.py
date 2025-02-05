#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from aepsych.benchmark.test_functions import f_1d, f_pairwise
from aepsych.generators import OptimizeAcqfGenerator, SobolGenerator
from aepsych.models import PairwiseProbitModel
from aepsych.strategy import SequentialStrategy, Strategy
from aepsych.transforms import (
    ParameterTransformedGenerator,
    ParameterTransformedModel,
    ParameterTransforms,
)
from aepsych.transforms.ops import NormalizeScale
from botorch.acquisition import qUpperConfidenceBound
from scipy.stats import bernoulli


class PairwiseProbitModelStrategyTest(unittest.TestCase):
    def test_1d_pairwise_probit(self):
        """
        test our 1d gaussian bump example
        """
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 50
        n_opt = 1
        lb = torch.tensor([-4.0])
        ub = torch.tensor([4.0])
        extra_acqf_args = {"beta": 3.84}
        transforms = ParameterTransforms(
            normalize=NormalizeScale(d=1, bounds=torch.stack([lb, ub]))
        )
        sobol_gen = ParameterTransformedGenerator(
            generator=SobolGenerator,
            lb=lb,
            ub=ub,
            seed=seed,
            stimuli_per_trial=2,
            transforms=transforms,
        )
        acqf_gen = ParameterTransformedGenerator(
            generator=OptimizeAcqfGenerator,
            acqf=qUpperConfidenceBound,
            acqf_kwargs=extra_acqf_args,
            stimuli_per_trial=2,
            transforms=transforms,
            lb=lb,
            ub=ub,
        )
        probit_model = ParameterTransformedModel(
            model=PairwiseProbitModel, lb=lb, ub=ub, transforms=transforms
        ).to("cuda")
        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                generator=sobol_gen,
                min_asks=n_init,
                stimuli_per_trial=2,
                outcome_types=["binary"],
                transforms=transforms,
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=probit_model,
                generator=acqf_gen,
                min_asks=n_opt,
                stimuli_per_trial=2,
                outcome_types=["binary"],
                transforms=transforms,
                use_gpu_generating=True,
                use_gpu_modeling=True,
            ),
        ]

        strat = SequentialStrategy(model_list)

        for _i in range(n_init + n_opt):
            next_pair = strat.gen().cpu()
            strat.add_data(
                next_pair, [bernoulli.rvs(f_pairwise(f_1d, next_pair, noise_scale=0.1))]
            )

        x = torch.linspace(-4, 4, 100)

        zhat, _ = strat.predict(x)

        self.assertTrue(np.abs(x[np.argmax(zhat.cpu().detach().numpy())]) < 0.5)
        self.assertTrue(strat.model.device.type == "cuda")
