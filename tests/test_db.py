#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import time
import unittest
import uuid
from configparser import DuplicateOptionError
from pathlib import Path

import aepsych.config as configuration
import aepsych.database.db as db
import aepsych.database.tables as tables
import pandas as pd
import sqlalchemy


class DBTestCase(unittest.TestCase):
    def setUp(self):
        # random datebase path name without dashes
        self._dbname = "./{}.db".format(str(uuid.uuid4().hex))
        self._database = db.Database(db_path=self._dbname)

    def tearDown(self):
        time.sleep(0.1)
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

        result = self._database.get_replay_for(master_table.unique_id)

        self.assertNotEqual(None, result)
        self.assertEqual(len(result), 1)
        self._database.record_message(
            master_table=master_table,
            type="test_type",
            request={"test": "this is a follow on request"},
        )

        result = self._database.get_replay_for(master_table.unique_id)
        self.assertNotEqual(None, result)
        self.assertEqual(len(result), 2)

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

        new_master = self._database.get_master_record(master_table.unique_id)
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

        # add sleep to ensure file is ready
        time.sleep(0.1)

        # open the new db
        test_database = db.Database(db_path=dst_db_path.as_posix(), update=False)

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

    def test_update_db_with_raw_data_tables(self):
        current_path = Path(os.path.abspath(__file__)).parent
        db_path = current_path
        db_path = db_path.joinpath("test_databases/multi_stimuli.db")

        # copy the db to a new file
        dst_db_path = Path(self._dbname)
        shutil.copy(str(db_path), str(dst_db_path))
        self.assertTrue(dst_db_path.is_file())

        # sleep to ensure db is ready
        time.sleep(0.1)

        # open the new db
        test_database = db.Database(db_path=dst_db_path.as_posix(), update=False)

        # Make sure that update is required
        self.assertTrue(test_database.is_update_required())

        # Update the database
        test_database.perform_updates()

        # Check that the update was successful

        # Known expected data
        par1 = [[0.1, 0.2], [0.3, 1], [2, 3], [4, 0.1], [0.2, 2], [1, 0.3], [0.3, 0.1]]
        par2 = [[4, 0.1], [3, 0.2], [2, 1], [0.3, 0.2], [2, 0.3], [1, 0.1], [0.3, 4]]
        outcomes = [[1, 0], [-1, 0], [0.1, 0], [0, 0], [-0.1, 0], [0, 0], [0, 0]]

        param_dict_expected = {x: {} for x in range(1, 8)}
        for i in range(1, 8):
            param_dict_expected[i]["par1_stimuli0"] = par1[i - 1][0]
            param_dict_expected[i]["par1_stimuli1"] = par1[i - 1][1]
            param_dict_expected[i]["par2_stimuli0"] = par2[i - 1][0]
            param_dict_expected[i]["par2_stimuli1"] = par2[i - 1][1]

        outcome_dict_expected = {x: {} for x in range(1, 8)}
        for i in range(1, 8):
            outcome_dict_expected[i]["outcome_0"] = outcomes[i - 1][0]
            outcome_dict_expected[i]["outcome_1"] = outcomes[i - 1][1]

        # Check that the number of entries in each table is correct
        n_iterations = (
            test_database.get_engine()
            .execute("SELECT COUNT(*) FROM raw_data")
            .fetchone()[0]
        )
        self.assertEqual(n_iterations, 7)
        n_params = (
            test_database.get_engine()
            .execute("SELECT COUNT(*) FROM param_data")
            .fetchone()[0]
        )
        self.assertEqual(n_params, 28)
        n_outcomes = (
            test_database.get_engine()
            .execute("SELECT COUNT(*) FROM outcome_data")
            .fetchone()[0]
        )
        self.assertEqual(n_outcomes, 14)

        # Check that the data is correct
        param_data = (
            test_database.get_engine().execute("SELECT * FROM param_data").fetchall()
        )
        param_dict = {x: {} for x in range(1, 8)}
        for param in param_data:
            param_dict[param.iteration_id][param.param_name] = float(param.param_value)

        self.assertEqual(param_dict, param_dict_expected)

        outcome_data = (
            test_database.get_engine().execute("SELECT * FROM outcome_data").fetchall()
        )
        outcome_dict = {x: {} for x in range(1, 8)}
        for outcome in outcome_data:
            outcome_dict[outcome.iteration_id][outcome.outcome_name] = (
                outcome.outcome_value
            )

        self.assertEqual(outcome_dict, outcome_dict_expected)

        # Check if we have the extra_data column
        pragma = (
            test_database.get_engine()
            .execute(
                "SELECT * FROM pragma_table_info('raw_data') WHERE name='extra_data'"
            )
            .fetchall()
        )
        self.assertTrue(len(pragma) == 1)

        # Make sure that update is no longer required
        self.assertFalse(test_database.is_update_required())

        test_database.delete_db()

    def test_update_db_with_raw_extra_data(self):
        current_path = Path(os.path.abspath(__file__)).parent
        db_path = current_path
        db_path = db_path.joinpath("test_databases/extra_info.db")

        # copy the db to a new file
        dst_db_path = Path(self._dbname)
        shutil.copy(str(db_path), str(dst_db_path))
        self.assertTrue(dst_db_path.is_file())

        # sleep to ensure db is ready
        time.sleep(0.1)

        # open the new db
        test_database = db.Database(db_path=dst_db_path.as_posix(), update=False)

        replay_tells = [
            row for row in test_database.get_replay_for(1) if row.message_type == "tell"
        ]

        # Make sure that update is required
        self.assertTrue(test_database.is_update_required())

        # Update the database
        test_database.perform_updates()

        # The trial numbers line up with tells
        none_rows = 0
        for row in test_database.get_raw_for(1):
            if row.extra_data is None:
                none_rows += 1
            else:
                self.assertTrue(row.unique_id == row.extra_data["trial_number"])
                self.assertTrue(row.extra_data["extra"] == "info")

        # Exactly one row should be none
        self.assertTrue(none_rows == 1)

        self.assertFalse(test_database.is_update_required())

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
        unique_id = master_table.unique_id
        strat = self._database.get_strat_for(unique_id)
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

        unique_id = master_table.unique_id

        config = self._database.get_config_for(unique_id)

        self.assertEqual(test_config, config)

    def test_raw_table(self):
        model_data = True
        master_table = self._database.record_setup(
            description="test raw table",
            name="test",
            request={"test": "this a test request"},
        )
        # Record a raw data entry
        self._database.record_raw(master_table, model_data=model_data)
        unique_id = master_table.unique_id
        raw_data = self._database.get_raw_for(unique_id)
        self.assertEqual(len(raw_data), 1)
        self.assertEqual(raw_data[0].model_data, model_data)

    def test_param_table(self):
        param_name = "test_param"
        param_value = 1.123
        master_table = self._database.record_setup(
            description="test param table",
            name="test",
            request={"test": "this a test request"},
        )
        raw_table = self._database.record_raw(master_table, model_data=True)
        # Record a param data entry
        self._database.record_param(raw_table, param_name, param_value)
        iteration_id = raw_table.unique_id
        param_data = self._database.get_params_for(iteration_id)
        self.assertEqual(len(param_data), 1)
        self.assertEqual(param_data[0][0].param_name, param_name)
        self.assertEqual(float(param_data[0][0].param_value), param_value)

    def test_outcome_table(self):
        outcome_value = 1.123
        outcome_name = "test_outcome"
        master_table = self._database.record_setup(
            description="test outcome table",
            name="test",
            request={"test": "this a test request"},
        )
        raw_table = self._database.record_raw(master_table, model_data=True)
        # Record an outcome data entry
        self._database.record_outcome(raw_table, outcome_name, outcome_value)
        iteration_id = raw_table.unique_id
        outcome_data = self._database.get_outcomes_for(iteration_id)
        self.assertEqual(len(outcome_data), 1)
        self.assertEqual(outcome_data[0][0].outcome_name, outcome_name)
        self.assertEqual(outcome_data[0][0].outcome_value, outcome_value)

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
        stimuli_per_trial = 1
        outcome_types = [binary]

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
        self._database.record_config(master_table, generated_config)
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
        summary = self._database.summarize_experiments()
        self.assertIn("metadata1", summary.columns)
        self.assertIn("metadata2", summary.columns)

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

    def test_summarize_experiments(self):
        current_path = Path(os.path.abspath(__file__)).parent
        db_path = current_path
        db_path = db_path.joinpath("test_databases/1000_outcome.db")
        data = db.Database(db_path)
        summary = data.summarize_experiments()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), len(data.get_master_records()))
        colnames = [
            "experiment_id",
            "experiment_name",
            "experiment_description",
            "participant_id",
            "creation_time",
            "time_last_modified",
            "stimuli_per_trial",
            "parameter_names",
            "outcome_names",
            "n_data",
        ]
        for col in colnames:
            self.assertTrue(col in summary.columns)
        self.assertTrue(
            (summary["creation_time"] < summary["time_last_modified"]).all()
        )
        self.assertTrue(pd.api.types.is_integer_dtype(summary["stimuli_per_trial"]))
        self.assertTrue(pd.api.types.is_integer_dtype(summary["n_data"]))

    def test_get_dataframe(self):
        current_path = Path(os.path.abspath(__file__)).parent
        db_path = current_path
        db_path = db_path.joinpath("test_databases/1000_outcome.db")
        data = db.Database(db_path)
        df = data.get_data_frame()
        self.assertIsInstance(df, pd.DataFrame)
        colnames = [
            "experiment_id",
            "experiment_name",
            "experiment_description",
            "participant_id",
            "timestamp",
            "par1",
            "par2",
            "outcome",
            "contPar",
            "par1_stimuli0",
            "par1_stimuli1",
            "par2_stimuli0",
            "par2_stimuli1",
        ]
        for col in colnames:
            self.assertTrue(col in df.columns)

        n = 0
        for rec in data.get_master_records():
            n += len(data.get_raw_for(rec.unique_id))
        self.assertEqual(n, len(df))


if __name__ == "__main__":
    unittest.main()
