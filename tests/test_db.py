#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import unittest
import uuid
from configparser import DuplicateOptionError
from pathlib import Path

import aepsych.config as configuration
import aepsych.database.db as db
import aepsych.database.tables as tables
import sqlalchemy


class DBTestCase(unittest.TestCase):
    def setUp(self):
        # random datebase path name without dashes
        self._dbname = "./{}.db".format(str(uuid.uuid4().hex))
        self._database = db.Database(db_path=self._dbname)

    def tearDown(self):
        self._database.delete_db()

    def test_db_create(self):
        engine = self._database.get_engine()
        self.assertIsNotNone(engine)
        self.assertIsNotNone(self._database._engine)

    def test_record_setup_basic(self):
        master_table = self._database.record_setup(
            description="test description",
            name="test name",
            request={"test": "this is a test request"},
        )

        result = self._database.get_replay_for(master_table.experiment_id)

        self.assertNotEqual(None, result)
        self.assertEqual(len(result), 1)
        self._database.record_message(
            master_table=master_table,
            type="test_type",
            request={"test": "this is a follow on request"},
        )

        result = self._database.get_replay_for(master_table.experiment_id)
        self.assertNotEqual(None, result)
        self.assertEqual(len(result), 2)

    def test_record_setup_doublesetup_goodid(self):
        master_table = self._database.record_setup(
            description="test description",
            name="test name",
            request={"test": "this is a test request"},
        )
        self.assertIsNotNone(master_table)
        self.assertEqual(len(master_table.children_replay), 1)
        master_table = self._database.record_setup(
            description="test description",
            name="test name",
            request={"test": "this is a test request"},
            id=master_table.experiment_id,
        )
        self.assertIsNotNone(master_table)
        self.assertEqual(len(master_table.children_replay), 2)

    def test_record_setup_doublesetup_badid(self):
        master_table = self._database.record_setup(
            description="test description",
            name="test name",
            request={"test": "this is a test request"},
        )
        self.assertIsNotNone(master_table)
        self.assertEqual(len(master_table.children_replay), 1)
        self.assertRaises(
            RuntimeError,
            self._database.record_setup,
            description="test description",
            name="test name",
            request={"test": "this is a test request"},
            id=1,
        )

    def test_record_setup_master_children(self):
        master_table = self._database.record_setup(
            description="test description",
            name="test name",
            request={"test": "this is a test request"},
        )
        self.assertIsNotNone(master_table)
        self.assertEqual(len(master_table.children_replay), 1)
        self._database.record_message(
            master_table, "test", request={"test": "this is a test request"}
        )
        self.assertEqual(len(master_table.children_replay), 2)

    def test_extra_info(self):
        extra_info_setup = {"test": "this is extra_info"}
        master_table = self._database.record_setup(
            description="test description",
            name="test name",
            request={"test": "this is a test request", "extra_info": extra_info_setup},
        )
        extra_info_record = {"test": "This is another extra_info"}
        self._database.record_message(
            master_table,
            "test",
            request={"test": "this is a test request", "extra_info": extra_info_record},
        )

        new_master = self._database.get_master_record(master_table.experiment_id)
        self.assertEqual(new_master.children_replay[0].extra_info, extra_info_setup)
        self.assertEqual(new_master.children_replay[1].extra_info, extra_info_record)

    def test_update_db(self):
        current_path = Path(os.path.abspath(__file__)).parent
        db_path = current_path
        db_path = db_path.joinpath("test_databases/test_original_schema.db")

        # copy the db to a new file
        dst_db_path = Path(self._dbname)
        shutil.copy(str(db_path), str(dst_db_path))
        self.assertTrue(dst_db_path.is_file())

        # open the new db
        test_database = db.Database(db_path=dst_db_path.as_posix())

        self.assertFalse(tables.DbReplayTable._has_extra_info(test_database._engine))
        self.assertTrue(test_database.is_update_required())

        # make sure we raise the exception on newer columns
        self.assertRaises(
            sqlalchemy.exc.OperationalError,
            test_database.record_setup,
            description="test description",
            name="test name",
            request={"test": "this is a test request"},
        )
        test_database._session.rollback()
        test_database.perform_updates()

        # retry adding rows
        master_table = test_database.record_setup(
            description="test description",
            name="test name",
            request={"test": "this is a test request"},
        )
        test_database.record_message(
            master_table, "test", request={"test": "this is a test request"}
        )
        # make sure the new column exists
        self.assertTrue(tables.DbReplayTable._has_extra_info(test_database._engine))

        test_database.delete_db()

    def test_update_configs(self):
        config_str = """
        [common]
        parnames = [par1, par2]
        lb = [0, 0]
        ub = [1, 1]
        outcome_type = single_probit
        target = 0.75

        [SobolStrategy]
        n_trials = 10

        [ModelWrapperStrategy]
        n_trials = 20
        refit_every = 5

        [experiment]
        acqf = MonotonicMCLSE
        init_strat_cls = SobolStrategy
        opt_strat_cls = ModelWrapperStrategy
        modelbridge_cls = MonotonicSingleProbitModelbridge
        model = MonotonicRejectionGP

        [MonotonicMCLSE]
        beta = 3.84

        [MonotonicRejectionGP]
        inducing_size = 100
        mean_covar_factory = monotonic_mean_covar_factory

        [MonotonicSingleProbitModelbridge]
        restarts = 10
        samps = 1000
        """

        request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": config_str},
        }

        dbname = "./{}.db".format(str(uuid.uuid4().hex))
        database = db.Database(dbname)
        database.record_setup(
            description="default description",
            name="default name",
            request=request,
        )

        self.assertTrue(database.is_update_required())
        database.perform_updates()
        self.assertFalse(database.is_update_required())
        database.delete_db()

    def test_strat_table(self):
        test_strat = {"strat": "this is nothing like a strat"}
        master_table = self._database.record_setup(
            description="test description",
            name="test name",
            request={"test": "this is a test request"},
        )
        # record a strat
        self._database.record_strat(master_table, strat=test_strat)
        experiment_id = master_table.experiment_id
        strat = self._database.get_strat_for(experiment_id)
        self.assertEqual(test_strat, strat)

    def test_config_table(self):
        test_config = {"config": "this is nothing like a config but it works."}
        master_table = self._database.record_setup(
            description="test description",
            name="test name",
            request={"test": "this is a test request"},
        )
        # record a strat
        self._database.record_config(master_table, config=test_config)

        experiment_id = master_table.experiment_id

        config = self._database.get_config_for(experiment_id)

        self.assertEqual(test_config, config)

    # Test some metadata flow stuff and see if it is working.
    def test_metadata(self):
        # Run tests using the native config_str functionality.
        config_str = """
        [common]
        parnames = [par1, par2]
        lb = [0, 0]
        ub = [1, 1]
        outcome_type = single_probit
        target = 0.75

        [SobolStrategy]
        n_trials = 10

        [ModelWrapperStrategy]
        n_trials = 20
        refit_every = 5

        [experiment]
        acqf = MonotonicMCLSE
        init_strat_cls = SobolStrategy
        opt_strat_cls = ModelWrapperStrategy
        modelbridge_cls = MonotonicSingleProbitModelbridge
        model = MonotonicRejectionGP

        [MonotonicMCLSE]
        beta = 3.98

        [MonotonicRejectionGP]
        inducing_size = 100
        mean_covar_factory = monotonic_mean_covar_factory

        [MonotonicSingleProbitModelbridge]
        restarts = 10
        samps = 1000

        [metadata]
        experiment_name = Lucas
        experiment_description = Test
        metadata1 = one
        metadata2 = two


        """

        request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": config_str},
        }
        # Generate a config for later to run .jsonifyMetadata() on.
        generated_config = configuration.Config(**request["message"])
        master_table = self._database.record_setup(
            description=generated_config["metadata"]["experiment_description"],
            name=generated_config["metadata"]["experiment_name"],
            request=request,
            extra_metadata=generated_config.jsonifyMetadata(),
        )
        self.assertEqual(
            generated_config.jsonifyMetadata(),
            master_table.extra_metadata,  # Test in JSON form
        )
        # Next I can deserialize into a dictionary and make sure each element is 1-to-1.
        ## Important thing to note is generated_config will have extra fields because of configparser's.
        ## Run comparison of json.loads -> generated_config, NOT the other way around.

        deserializedjson = json.loads(
            master_table.extra_metadata
        )  # Directly from master table entry.

        ## Going to check each value in the deserialized json from the DB to the expected values along with the config prior to insertion.
        ## This will check if it retains the individual values.
        self.assertEqual(deserializedjson["metadata1"], "one")
        self.assertEqual(deserializedjson["metadata2"], "two")
        self.assertEqual(deserializedjson["experiment_name"], "Lucas")
        self.assertEqual(deserializedjson["experiment_description"], "Test")
        self.assertEqual(
            deserializedjson["experiment_name"], master_table.experiment_name
        )
        self.assertEqual(
            deserializedjson["experiment_description"],
            master_table.experiment_description,
        )

    def test_broken_metadata(self):
        # We are going to be testing some broken metadata here. We need to make sure it does not misbehave.
        config_strdupe = """
        [common]
        parnames = [par1, par2]
        lb = [0, 0]
        ub = [1, 1]
        outcome_type = single_probit
        target = 0.75

        [SobolStrategy]
        n_trials = 10

        [ModelWrapperStrategy]
        n_trials = 20
        refit_every = 5

        [experiment]
        acqf = MonotonicMCLSE
        init_strat_cls = SobolStrategy
        opt_strat_cls = ModelWrapperStrategy
        modelbridge_cls = MonotonicSingleProbitModelbridge
        model = MonotonicRejectionGP

        [MonotonicMCLSE]
        beta = 3.98

        [MonotonicRejectionGP]
        inducing_size = 100
        mean_covar_factory = monotonic_mean_covar_factory

        [MonotonicSingleProbitModelbridge]
        restarts = 10
        samps = 1000

        [metadata]
        experiment_name = Lucas
        experiment_description = Test
        metadata1 =
        metadata2 = two
        metadata2 = three


        """

        config_str = """
        [common]
        parnames = [par1, par2]
        lb = [0, 0]
        ub = [1, 1]
        outcome_type = single_probit
        target = 0.75

        [SobolStrategy]
        n_trials = 10

        [ModelWrapperStrategy]
        n_trials = 20
        refit_every = 5

        [experiment]
        acqf = MonotonicMCLSE
        init_strat_cls = SobolStrategy
        opt_strat_cls = ModelWrapperStrategy
        modelbridge_cls = MonotonicSingleProbitModelbridge
        model = MonotonicRejectionGP

        [MonotonicMCLSE]
        beta = 3.98

        [MonotonicRejectionGP]
        inducing_size = 100
        mean_covar_factory = monotonic_mean_covar_factory

        [MonotonicSingleProbitModelbridge]
        restarts = 10
        samps = 1000

        [metadata]
        metadata1 =
        metadata2 = three


        """

        request = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": config_strdupe},
        }
        request2 = {
            "type": "setup",
            "version": "0.01",
            "message": {"config_str": config_str},
        }
        # Generate a config for later to run .jsonifyMetadata() on.
        with self.assertRaises(DuplicateOptionError):
            configuration.Config(**request["message"])
        generated_config = configuration.Config(**request2["message"])

        master_table = self._database.record_setup(
            description=(
                generated_config["metadata"]["experiment_description"]
                if ("experiment_description" in generated_config["metadata"].keys())
                else "default description"
            ),
            name=(
                generated_config["metadata"]["experiment_name"]
                if ("experiment_name" in generated_config["metadata"].keys())
                else "default name"
            ),
            request=request,
            extra_metadata=generated_config.jsonifyMetadata(),
        )
        deserializedjson = json.loads(
            master_table.extra_metadata
        )  # This is initial process is exactly the same but now we switch things up...
        self.assertEqual(deserializedjson["metadata2"], "three")  # test normal value
        self.assertEqual(deserializedjson["metadata1"], "")  # test an empty value
        self.assertEqual(
            master_table.experiment_name, "default name"
        )  # test default name value
        self.assertEqual(
            master_table.experiment_description, "default description"
        )  # test default description value


if __name__ == "__main__":
    unittest.main()
