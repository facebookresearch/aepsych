#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from aepsych.config import Config, ParameterConfigError
from aepsych.generators import SobolGenerator
from aepsych.models import GPClassificationModel
from aepsych.strategy import SequentialStrategy
from aepsych.transforms import (
    ParameterTransformedGenerator,
    ParameterTransformedModel,
    ParameterTransforms,
)
from aepsych.transforms.ops import Fixed, Log10Plus, NormalizeScale, Round


class TransformsWrapperTest(unittest.TestCase):
    def test_model_mode_change(self):
        transforms = ParameterTransforms(norm=NormalizeScale(d=3))
        model = ParameterTransformedModel(
            GPClassificationModel, dim=3, transforms=transforms
        )

        # Starts both in training
        self.assertTrue(model.training)
        self.assertTrue(model.transforms.training)

        # Swap to eval
        model.eval()
        self.assertFalse(model.training)
        self.assertFalse(model.transforms.training)

        # Swap back to train
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.transforms.training)

    def test_generator_mode_change(self):
        transforms = ParameterTransforms(norm=NormalizeScale(d=3))
        generator = ParameterTransformedGenerator(
            SobolGenerator,
            lb=torch.tensor([0, 0, 0]),
            ub=torch.tensor([1, 1, 1]),
            transforms=transforms,
        )

        # Starts both in training
        with self.assertWarns(
            Warning
        ):  # Sobol can't be moved to eval, so it should warn
            self.assertTrue(generator.training)
            self.assertTrue(generator.transforms.training)

            # Swap to eval
            generator.eval()
            self.assertFalse(generator.training)
            self.assertFalse(generator.transforms.training)

            # Swap back to train
            generator.train()
            self.assertTrue(generator.training)
            self.assertTrue(generator.transforms.training)


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
            lb=torch.tensor([1.0, 1.0]),
            ub=torch.tensor([100.0, 100.0]),
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
        config_generator = self.strat.strat_list[1].generator

        obj_model = ParameterTransformedModel(
            model=GPClassificationModel,
            transforms=self.strat.strat_list[1].transforms,
            dim=2,
        )
        obj_generator = ParameterTransformedGenerator(
            generator=SobolGenerator,
            lb=torch.tensor([1.0, 1.0]),
            ub=torch.tensor([100.0, 100.0]),
            transforms=self.strat.strat_list[1].transforms,
        )

        self.assertTrue(type(config_model._base_obj) is type(obj_model._base_obj))
        self.assertTrue(torch.equal(config_generator.lb, obj_generator.lb))
        self.assertTrue(torch.equal(config_generator.ub, obj_generator.ub))
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

    def test_transform_manual_generator(self):
        base_points = [
            [[-1.5, 1], [-1, 1.25], [-2, 1.75]],
            [[-1.25, 1.25], [-1.75, 1.5], [-1.0, 2]],
        ]
        window = [0.25, 0.1]
        samples_per_point = 2
        lb = [-3, 1]
        ub = [-1, 3]
        config_str = f"""
            [common]
            parnames = [par1, par2]
            stimuli_per_trial = 3
            outcome_types = [binary]
            strategy_names = [init_strat]

            [par1]
            par_type = continuous
            lower_bound = {lb[0]}
            upper_bound = {ub[0]}

            [par2]
            par_type = continuous
            lower_bound = {lb[1]}
            upper_bound = {ub[1]}

            [init_strat]
            generator = SampleAroundPointsGenerator

            [SampleAroundPointsGenerator]
            points = {base_points}
            window = {window}
            samples_per_point = {samples_per_point}
            seed = 123
        """
        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)

        nPoints = 0
        while not strat.finished:
            points = strat.gen()
            strat.add_data(points, torch.tensor(1.0))
            self.assertTrue(torch.all(points[0, 0, :] < 0))
            self.assertTrue(torch.all(points[0, 1, :] > 0))
            nPoints += 1

        self.assertTrue(nPoints == len(base_points) * samples_per_point)


class TransformsLog10Test(unittest.TestCase):
    def test_transform_reshape3D(self):
        lb = torch.tensor([-1.0, 0.0, 10.0])
        ub = torch.tensor([-1e-6, 9.0, 99.0])
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


class TransformsInteger(unittest.TestCase):
    def test_integer_bounds(self):
        config_str = """
            [common]
            parnames = [signal1, signal2]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat]

            [signal1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [signal2]
            par_type = integer
            lower_bound = 1
            upper_bound = 5

            [init_strat]
            generator = SobolGenerator
            min_asks = 1
        """
        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)
        points = strat.gen()[0]

        self.assertTrue((points[0] % 1).item() != 0.0)
        self.assertTrue((points[1] % 1).item() == 0.0)
        self.assertTrue(torch.all(strat._strat.generator.lb == 0))
        self.assertTrue(torch.all(strat._strat.generator.ub == 1))

        bad_config_str = """
            [common]
            parnames = [signal1, signal2]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat]

            [signal1]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [signal2]
            par_type = discrete
            lower_bound = 1
            upper_bound = 4.5

            [init_strat]
            generator = SobolGenerator
            min_asks = 1
        """
        config = Config()
        with self.assertRaises(ParameterConfigError):
            config.update(config_str=bad_config_str)

    def test_integer_model(self):
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
            par_type = integer
            lower_bound = {lower_bound}
            upper_bound = {upper_bound}

            [init_strat]
            generator = SobolGenerator
            min_total_tells = 50

            [SobolGenerator]
            seed = 1

            [opt_strat]
            generator = OptimizeAcqfGenerator
            acqf = MCLevelSetEstimation
            model = GPClassificationModel
            min_total_tells = 1
        """

        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)

        while not strat.finished:
            next_x = strat.gen()
            self.assertTrue((next_x % 1).item() == 0.0)
            response = int(np.random.rand() < (next_x / 100))
            strat.add_data(next_x, [response])

        x = torch.linspace(lower_bound, upper_bound, 100)

        zhat, _ = strat.predict(x)
        est_max = x[np.argmin((zhat - target) ** 2)]
        diff = np.abs(est_max / 100 - target)
        self.assertTrue(diff < 0.15, f"Diff = {diff}")

    def test_binary(self):
        config_str = """
            [common]
            parnames = [signal1]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat]

            [signal1]
            par_type = binary

            [init_strat]
            generator = SobolGenerator
            min_asks = 1
        """
        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)

        transforms = strat.transforms

        self.assertTrue(len(transforms) == 1)
        self.assertTrue(isinstance(list(transforms.values())[0], Round))
        self.assertTrue(
            torch.all(config.gettensor("common", "lb") == torch.tensor([0]))
        )
        self.assertTrue(
            torch.all(config.gettensor("common", "ub") == torch.tensor([1]))
        )

        bad_config_str = """
            [common]
            parnames = [signal1]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat]

            [signal1]
            par_type = binary
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = SobolGenerator
            min_asks = 1
        """
        config = Config()

        with self.assertRaises(ParameterConfigError):
            config.update(config_str=bad_config_str)


class TransformsFixed(unittest.TestCase):
    def test_fixed_from_config(self):
        np.random.seed(1)
        torch.manual_seed(1)

        config_str = """
            [common]
            parnames = [signal1, signal2, signal3, signal4]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat, opt_strat]

            [signal1]
            par_type = binary

            [signal2]
            par_type = fixed
            value = 4.5

            [signal3]
            par_type = continuous
            lower_bound = 1
            upper_bound = 100
            log_scale = True

            [signal4]
            par_type = fixed
            value = blue

            [init_strat]
            generator = SobolGenerator
            min_asks = 1

            [opt_strat]
            generator = OptimizeAcqfGenerator
            acqf = MCLevelSetEstimation
            model = GPClassificationModel
            min_asks = 1
        """
        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)

        while not strat.finished:
            points = strat.gen()
            self.assertTrue(points[0][1].item() == 4.5)
            strat.add_data(points, int(np.random.rand() > 0.5))

        self.assertTrue(len(strat.strat_list[0].generator.lb) == 2)
        self.assertTrue(len(strat.strat_list[0].generator.ub) == 2)
        self.assertTrue(strat.strat_list[-1].model.dim == 2)

        bad_config_str = """
            [common]
            parnames = [signal1, signal2, signal3]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat, opt_strat]

            [signal1]
            par_type = binary

            [signal2]
            par_type = fixed

            [signal3]
            par_type = continuous
            lower_bound = 1
            upper_bound = 100
            log_scale = True

            [init_strat]
            generator = SobolGenerator
            min_asks = 1

            [opt_strat]
            generator = OptimizeAcqfGenerator
            acqf = MCLevelSetEstimation
            model = GPClassificationModel
            min_asks = 1
        """
        config = Config()
        with self.assertRaises(ParameterConfigError):
            config.update(config_str=bad_config_str)

    def test_fixed_standalone(self):
        fixed1 = Fixed([3], values=[0.3])
        fixed2 = Fixed([1, 2], values=[0.1, 0.2])

        transforms = ParameterTransforms(fixed1=fixed1, fixed2=fixed2)

        self.assertTrue(len(transforms) == 1)
        self.assertTrue(
            torch.all(transforms["_CombinedFixed"].indices == torch.tensor([1, 2, 3]))
        )
        self.assertTrue(
            torch.all(
                transforms["_CombinedFixed"].values == torch.tensor([0.1, 0.2, 0.3])
            )
        )

        input = torch.tensor([[1, 100, 100, 100, 1], [2, 100, 100, 100, 2]])
        transformed = transforms.transform(input)
        untransformed = transforms.untransform(transformed)

        self.assertTrue(transformed.shape[0] == 2)
        self.assertTrue(torch.all(transformed[:, 0] == torch.tensor([1, 2])))
        self.assertTrue(
            torch.all(torch.tensor([1, 0.1, 0.2, 0.3, 1]) == untransformed[0])
        )
        self.assertTrue(
            torch.all(torch.tensor([2, 0.1, 0.2, 0.3, 2]) == untransformed[1])
        )

    def test_fixed_conflict(self):
        fixed1 = Fixed([3], values=[0], string_map={3: ["blue"]})
        fixed2 = Fixed(
            [1, 2], values=[0, 0.2], string_map={1: ["red"], 3: ["green", "blue"]}
        )

        with self.assertRaises(RuntimeError):
            transforms = ParameterTransforms(fixed1=fixed1, fixed2=fixed2)
