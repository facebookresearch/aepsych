#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import unittest
import uuid

import numpy as np
import torch
from aepsych import server, utils_logging
from aepsych.config import Config
from aepsych.generators import SobolGenerator
from aepsych.models import GPClassificationModel
from aepsych.server.message_handlers.handle_ask import ask
from aepsych.server.message_handlers.handle_setup import configure
from aepsych.server.message_handlers.handle_tell import tell
from aepsych.strategy import SequentialStrategy
from aepsych.transforms import (
    ParameterTransformedGenerator,
    ParameterTransformedModel,
    ParameterTransforms,
)
from aepsych.transforms.parameters import Categorical, Log10Plus, NormalizeScale


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


class TransformInteger(unittest.TestCase):
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


class TransformCategorical(unittest.TestCase):
    def test_categorical_model(self):
        np.random.seed(1)
        torch.manual_seed(1)

        n_init = 50
        n_opt = 1
        target = 0.75
        config_str = f"""
            [common]
            parnames = [signal1, signal2]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat, opt_strat]
            target = {target}

            [signal1]
            par_type = categorical
            choices = [red, green, blue]

            [signal2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = SobolGenerator
            min_asks = {n_init}

            [opt_strat]
            generator = OptimizeAcqfGenerator
            acqf = MCLevelSetEstimation
            model = GPClassificationModel
            min_asks = {n_opt}
        """
        config = Config()
        config.update(config_str=config_str)

        strat = SequentialStrategy.from_config(config)
        transforms = strat.transforms
        while not strat.finished:
            points = strat.gen()
            points = transforms.indices_to_str(points)

            if points[0][0] == "blue":
                response = int(np.random.rand() < points[0][1])
            else:
                response = 0

            strat.add_data(transforms.str_to_indices(points), response)

        _, loc = strat.model.get_max()
        loc = transforms.indices_to_str(loc)[0]

        self.assertTrue(loc[0] == "blue")
        self.assertTrue(loc[1] - target < 0.15)

    def test_standalone_transform(self):
        categorical_map = {1: ["red", "green", "blue"], 3: ["big", "small"]}
        input = torch.tensor([[0.2, 2, 4, 0, 1], [0.5, 0, 3, 0, 1], [0.9, 1, 0, 1, 0]])
        input_cats = np.array(
            [
                [0.2, "blue", 4, "big", "right"],
                [0.5, "red", 3, "big", "right"],
                [0.9, "green", 0, "small", "left"],
            ],
            dtype="O",
        )

        transforms = ParameterTransforms(
            categorical1=Categorical([1, 3], categorical_map=categorical_map),
            categorical2=Categorical([4], categorical_map={4: ["left", "right"]}),
        )

        self.assertTrue("_CombinedCategorical" in list(transforms.keys()))
        self.assertTrue("categorical1" not in list(transforms.keys()))

        transformed = transforms.transform(input)
        untransformed = transforms.untransform(transformed)

        self.assertTrue(torch.equal(input, untransformed))

        strings = transforms.indices_to_str(input)
        self.assertTrue(np.all(input_cats == strings))

        indices = transforms.str_to_indices(input_cats)
        self.assertTrue(torch.all(indices == input))


class TransformServer(unittest.TestCase):
    def setUp(self):
        # setup logger
        server.logger = utils_logging.getLogger(logging.DEBUG, "logs")
        # random datebase path name without dashes
        database_path = "./{}.db".format(str(uuid.uuid4().hex))
        self.s = server.AEPsychServer(database_path=database_path)

    def tearDown(self):
        self.s.cleanup()

        # cleanup the db
        if self.s.db is not None:
            self.s.db.delete_db()

    def test_categorical_smoketest(self):
        server = self.s
        config_str = f"""
            [common]
            parnames = [signal1, signal2]
            stimuli_per_trial = 1
            outcome_types = [binary]
            strategy_names = [init_strat, opt_strat]
            target = 0.75

            [signal1]
            par_type = categorical
            choices = [red, green, blue]

            [signal2]
            par_type = continuous
            lower_bound = 0
            upper_bound = 1

            [init_strat]
            generator = SobolGenerator
            min_asks = 1

            [opt_strat]
            generator = OptimizeAcqfGenerator
            acqf = MCLevelSetEstimation
            model = GPClassificationModel
            min_asks = 1
        """
        configure(
            server,
            config_str=config_str,
        )

        for _ in range(2):
            next_config = ask(server)

            self.assertTrue(isinstance(next_config["signal1"][0], str))

            tell(server, config=next_config, outcome=0)

    def test_pairwise_categorical(self):
        server = self.s
        config_str = """
            [common]
            stimuli_per_trial=2
            outcome_types=[binary]
            parnames = [x, y, z]
            strategy_names = [init_strat, opt_strat]
            
            [x]
            par_type = continuous
            lower_bound = 1
            upper_bound = 4
            normalize_scale = False

            [y]
            par_type = categorical
            choices = [red, green, blue]

            [z]
            par_type = discrete
            lower_bound = 1
            upper_bound = 1000
            log_scale = True

            [init_strat]
            min_asks = 1
            generator = SobolGenerator

            [opt_strat]
            model = PairwiseProbitModel
            min_asks = 1
            generator = OptimizeAcqfGenerator
            acqf = PairwiseMCPosteriorVariance

            [PairwiseProbitModel]
            mean_covar_factory = default_mean_covar_factory

            [PairwiseMCPosteriorVariance]
            objective = ProbitObjective
        """
        configure(server, config_str=config_str)

        for _ in range(2):
            next_config = ask(server)
            self.assertTrue(all([isinstance(val, str) for val in next_config["y"]]))
            tell(server, config=next_config, outcome=0)
