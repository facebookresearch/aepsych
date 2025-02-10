#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import unittest
import uuid

import numpy as np
import torch
from aepsych.acquisition import MCLevelSetEstimation
from aepsych.config import Config
from aepsych.generators import AsyncGenerator, OptimizeAcqfGenerator, SobolGenerator
from aepsych.models import GPClassificationModel
from aepsych.server import AEPsychBackgroundServer, AEPsychServer
from aepsych.strategy import SequentialStrategy
from aepsych_client import AEPsychClient
from sklearn.datasets import make_classification


class TestAsyncGenerator(unittest.TestCase):
    def test_timeout_fallback(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        timeout = 0.1

        X, y = make_classification(
            n_samples=100,
            n_features=8,
            n_redundant=3,
            n_informative=5,
            random_state=1,
            n_clusters_per_class=4,
        )
        X, y = torch.Tensor(X), torch.Tensor(y)
        lb = -3 * torch.ones(8)
        ub = 3 * torch.ones(8)
        inducing_size = 10

        model = GPClassificationModel(
            dim=8,
            inducing_size=inducing_size,
        )

        model.fit(X, y)
        primary_gen = OptimizeAcqfGenerator(
            acqf=MCLevelSetEstimation,
            lb=lb,
            ub=ub,
        )
        # Bounds are higher than real bounds to test fallback
        fallback_gen = SobolGenerator(lb=10 * torch.ones(8), ub=20 * torch.ones(8))
        generator = AsyncGenerator(primary_gen, fallback_gen, timeout=timeout)

        # A dummy call to make sure the generator is initialized
        points = generator.gen(1, model)

        start = time.time()
        points = generator.gen(1, model)
        end = time.time()
        fallback_time = end - start

        # Really loose test due to CI slowness
        self.assertTrue(fallback_time < 0.3, f"fallback_time: {fallback_time}")
        self.assertTrue(torch.all(points > 5))

        # Wait a bit so that the primary generator is ready
        time.sleep(10)

        start = time.time()
        points = generator.gen(1, model)
        end = time.time()
        gen_time = end - start
        self.assertTrue(gen_time < 0.01)
        self.assertTrue(torch.all(points < 5))

        # Delete it in case it is still running
        del generator

    def test_async_pickle(self):
        db_path = "./{}_test_server.db".format(str(uuid.uuid4().hex))
        # Create a server
        server = AEPsychBackgroundServer(host="127.0.0.1", database_path=db_path)
        server.start()
        time.sleep(1)

        # Make a client
        try_again = True
        attempts = 0
        while try_again:
            try_again = False
            attempts += 1
            try:
                client = AEPsychClient(ip=server.host, port=server.port)
            except ConnectionRefusedError:
                if attempts > 10:
                    raise ConnectionRefusedError
                try_again = True
                time.sleep(1)

        n_init = 10
        n_opt = 5
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

            [init_strat]
            generator = SobolGenerator
            min_asks = {n_init}

            [SobolGenerator]
            seed = 1

            [opt_strat]
            generator = AsyncGenerator
            model = GPClassificationModel
            min_asks = {n_opt}

            [AsyncGenerator]
            generator = OptimizeAcqfGenerator
            backup_generator = SobolGenerator
            timeout = 1

            [OptimizeAcqfGenerator]
            acqf = MCLevelSetEstimation
            """
        client.configure(config_str=config_str)

        finished = False
        signals = []
        responses = []
        while not finished:
            ask = client.ask()
            outcome = int(np.random.rand() < (ask["config"]["signal1"][0] / 100))
            client.tell(ask["config"], outcome)

            finished = ask["is_finished"]
            signals.append(ask["config"]["signal1"][0])
            responses.append(outcome)

        server.stop()

        server = AEPsychServer(database_path=db_path)
        unique_id = server.db.get_master_records()[-1].unique_id
        out_df = server.get_dataframe_from_replay(unique_id)

        self.assertTrue((out_df["response"] == responses).all())
        self.assertTrue((out_df["signal1"] == signals).all())

        server.db.delete_db()
        time.sleep(0.1)
