#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import numpy as np
import torch
from aepsych.config import Config
from aepsych.generators import SobolGenerator
from aepsych.models import GPClassificationModel
from aepsych.strategy import SequentialStrategy
from aepsych.transforms import (
    ParameterTransformedGenerator,
    ParameterTransformedModel,
    ParameterTransforms,
)
from aepsych.transforms.parameters import Log10Plus, NormalizeScale


class TransformsConfigTest(unittest.TestCase):
    def setUp(self):
        config_str = """
            [common]
            parnames = [signal1, signal2]
            stimuli_per_trial = 1
            outcome_types = [binary]
            target = 0.75
            strategy_names = [init_strat, opt_strat]

            [signal1]
            par_type = continuous
            lower_bound = 1
            upper_bound = 100
            log_scale = false

            [signal2]
            par_type = continuous
            lower_bound = 1
            upper_bound = 100
            log_scale = true

            [init_strat]
            min_total_tells = 10
            generator = SobolGenerator

            [SobolGenerator]
            seed = 12345

            [opt_strat]
            min_total_tells = 50
            refit_every = 5
            generator = OptimizeAcqfGenerator
            acqf = MCLevelSetEstimation
            model = GPClassificationModel
            """

        config = Config()
        config.update(config_str=config_str)

        self.strat = SequentialStrategy.from_config(config)

    def test_generator_init_equivalent(self):
        config_gen = self.strat.strat_list[0].generator

        class_gen = ParameterTransformedGenerator(
            generator=SobolGenerator,
            lb=torch.tensor([1, 1]),
            ub=torch.tensor([100, 100]),
            seed=12345,
            transforms=self.strat.strat_list[0].transforms,
        )

        self.assertTrue(type(config_gen._base_obj) is type(class_gen._base_obj))
        self.assertTrue(torch.equal(config_gen.lb, class_gen.lb))
        self.assertTrue(torch.equal(config_gen.ub, class_gen.ub))

        config_points = config_gen.gen(10)
        obj_points = class_gen.gen(10)
        self.assertTrue(torch.equal(config_points, obj_points))

        self.assertEqual(
            len(config_gen.transforms.values()), len(class_gen.transforms.values())
        )

    def test_model_init_equivalent(self):
        config_model = self.strat.strat_list[1].model

        obj_model = ParameterTransformedModel(
            model=GPClassificationModel,
            lb=torch.tensor([1, 1]),
            ub=torch.tensor([100, 100]),
            transforms=self.strat.strat_list[1].transforms,
        )

        self.assertTrue(type(config_model._base_obj) is type(obj_model._base_obj))
        self.assertTrue(torch.equal(config_model.bounds, obj_model.bounds))
        self.assertTrue(torch.equal(config_model.bounds, obj_model.bounds))

        self.assertEqual(
            len(config_model.transforms.values()), len(obj_model.transforms.values())
        )

    def test_transforms_in_strategy(self):
        for _strat in self.strat.strat_list:
            # Check if the same transform is passed around everywhere
            self.assertTrue(id(_strat.transforms) == id(_strat.generator.transforms))
            if _strat.model is not None:
                self.assertTrue(
                    id(_strat.generator.transforms) == id(_strat.model.transforms)
                )

            # Check all the transform bits are the same
            for strat_transform, gen_transform in zip(
                _strat.transforms.items(), _strat.generator.transforms.items()
            ):
                self.assertTrue(strat_transform[0] == gen_transform[0])
                self.assertTrue(type(strat_transform[1]) is type(gen_transform[1]))

    def test_options_override(self):
        config_str = """
            [common]
            parnames = [signal1, signal2]
            stimuli_per_trial = 1
            outcome_types = [binary]
            target = 0.75

            [signal1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1
            log_scale = false

            [signal2]
            par_type = continuous
            lower_bound = 1
            upper_bound = 100
            log_scale = True
            """
        config = Config()
        config.update(config_str=config_str)

        override = {
            "indices": [0],
            "constant": 5,
        }
        transform = Log10Plus.from_config(config, "signal2", override)

        self.assertTrue(transform.constant == 5)
        self.assertTrue(transform.indices[0] == 0)


class TransformsLog10Test(unittest.TestCase):
    def test_transform_reshape3D(self):
        lb = torch.tensor([-1, 0, 10])
        ub = torch.tensor([-1e-6, 9, 99])
        x = SobolGenerator(lb=lb, ub=ub, stimuli_per_trial=2).gen(4)

        transforms = ParameterTransforms(
            log10=Log10Plus(indices=[0, 1, 2], constant=2),
            normalize=NormalizeScale(d=3, bounds=torch.stack([lb, ub])),
        )

        transformed_x = transforms.transform(x)
        untransformed_x = transforms.untransform(transformed_x)

        self.assertTrue(torch.allclose(x, untransformed_x))

    def test_log_transform(self):
        config_str = """
            [common]
            parnames = [signal1, signal2]
            stimuli_per_trial = 1
            outcome_types = [binary]

            [signal1]
            par_type = continuous
            lower_bound = -10
            upper_bound = 10
            log_scale = false
            normalize_scale = no

            [signal2]
            par_type = continuous
            lower_bound = 1
            upper_bound = 100
            log_scale = true
            normalize_scale = off
        """
        config = Config()
        config.update(config_str=config_str)

        transforms = ParameterTransforms.from_config(config)

        values = torch.tensor([[1, 100], [-2, 10], [-3.2, 1]])
        expected = torch.tensor([[1, 2], [-2, 1], [-3.2, 0]])
        transformed = transforms.transform(values)

        self.assertTrue(torch.allclose(transformed, expected))
        self.assertTrue(torch.allclose(transforms.untransform(transformed), values))

    def test_log10Plus_transform(self):
        config_str = """
            [common]
            parnames = [signal1]
            stimuli_per_trial = 1
            outcome_types = [binary]

            [signal1]
            par_type = continuous
            lower_bound = -1
            upper_bound = 1
            log_scale = on
        """
        config = Config()
        config.update(config_str=config_str)

        transforms = ParameterTransforms.from_config(config)

        values = torch.tensor([[-1, 0, 0.5, 0.1]]).T
        transformed = transforms.transform(values)
        untransformed = transforms.untransform(transformed)

        self.assertTrue(torch.all(transformed >= 0))
        self.assertTrue(torch.allclose(values, untransformed))

    def test_log_model(self):
        np.random.seed(1)
        torch.manual_seed(1)

        lower_bound = 1
        upper_bound = 100
        target = 0.75

        config_str = f"""
            [common]
            parnames = [signal1]
            stimuli_per_trial = 1
            outcome_types = [binary]
            target = {target}
            strategy_names = [init_strat, opt_strat]

            [signal1]
            par_type = continuous
            lower_bound = {lower_bound}
            upper_bound = {upper_bound}
            log_scale = true

            [init_strat]
            generator = SobolGenerator
            min_total_tells = 50

            [SobolGenerator]
            seed = 1

            [opt_strat]
            generator = OptimizeAcqfGenerator
            acqf = MCLevelSetEstimation
            model = GPClassificationModel
            min_total_tells = 70
            """

        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)

        while not strat.finished:
            next_x = strat.gen()
            response = int(np.random.rand() < (next_x / 100))
            strat.add_data(next_x, [response])

        x = torch.linspace(lower_bound, upper_bound, 100)

        zhat, _ = strat.predict(x)
        est_max = x[np.argmin((zhat - target) ** 2)]
        diff = np.abs(est_max / 100 - target)
        self.assertTrue(diff < 0.15, f"Diff = {diff}")


class TransformsNormalize(unittest.TestCase):
    def test_normalize_scale(self):
        config_str = """
            [common]
            parnames = [signal1, signal2]
            stimuli_per_trial = 1
            outcome_types = [binary]

            [signal1]
            par_type = continuous
            lower_bound = -10
            upper_bound = 10
            normalize_scale = false

            [signal2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 100
        """
        config = Config()
        config.update(config_str=config_str)

        transforms = ParameterTransforms.from_config(config)

        values = torch.tensor([[-5.0, 20.0], [20.0, 1.0]])
        expected = torch.tensor([[-5.0, 0.2], [20.0, 0.01]])
        transformed = transforms.transform(values)

        self.assertTrue(torch.allclose(transformed, expected))
        self.assertTrue(torch.allclose(transforms.untransform(transformed), values))
