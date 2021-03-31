import unittest
import aepsych.database.db as db
import aepsych.database.tables as tables
from pathlib import Path
import os
import shutil
import sqlalchemy

class DBTestCase(unittest.TestCase):
    def setUp(self):
        self._dbname= "./test_default.db"
        self._database = db.Database(db_path=self._dbname)

    def tearDown(self):
        self._database.delete_db()

    def test_db_create(self):
        engine = self._database.get_engine()
        self.assertIsNotNone(engine)
        self.assertIsNotNone(self._database._engine)

    def test_record_setup_basic(self):
        master_table = self._database.record_setup(description="test description", name="test name", request={"test":"this is a test request"})

        result = self._database.get_replay_for(master_table.experiment_id)

        self.assertNotEqual(None, result)
        self.assertEqual(len(result), 1)
        self._database.record_message(master_table=master_table, type="test_type", request={"test":"this is a follow on request"})

        result = self._database.get_replay_for(master_table.experiment_id)
        self.assertNotEqual(None, result)
        self.assertEqual(len(result), 2)

    def test_record_setup_doublesetup_goodid(self):
        master_table = self._database.record_setup(description="test description", name="test name", request={"test":"this is a test request"})
        self.assertIsNotNone(master_table)
        self.assertEqual(len(master_table.children_replay), 1)
        master_table = self._database.record_setup(description="test description", name="test name", request={"test":"this is a test request"}, id=master_table.experiment_id)
        self.assertIsNotNone(master_table)
        self.assertEqual(len(master_table.children_replay), 2)

    def test_record_setup_doublesetup_badid(self):
        master_table = self._database.record_setup(description="test description", name="test name", request={"test":"this is a test request"})
        self.assertIsNotNone(master_table)
        self.assertEqual(len(master_table.children_replay), 1)
        self.assertRaises(RuntimeError, self._database.record_setup, description="test description", name="test name", request={"test":"this is a test request"}, id=1)


    def test_record_setup_master_children(self):
        master_table = self._database.record_setup(description="test description", name="test name", request={"test":"this is a test request"})
        self.assertIsNotNone(master_table)
        self.assertEqual(len(master_table.children_replay), 1)
        self._database.record_message(master_table, "test", request={"test":"this is a test request"})
        self.assertEqual(len(master_table.children_replay), 2)

    def test_extra_info(self):
        extra_info_setup = {"test":"this is extra_info"}
        master_table = self._database.record_setup(description="test description", name="test name", request={"test":"this is a test request", "extra_info": extra_info_setup})
        extra_info_record = {"test":"This is another extra_info"}
        self._database.record_message(master_table, "test", request={"test":"this is a test request", "extra_info": extra_info_record})

        new_master = self._database.get_master_record(master_table.experiment_id)
        self.assertEqual(new_master.children_replay[0].extra_info, extra_info_setup)
        self.assertEqual(new_master.children_replay[1].extra_info, extra_info_record)

    def test_update_db(self):
        current_path = Path(os.path.abspath(__file__)).parent
        db_path = current_path
        db_path = db_path.joinpath("test_databases/test_original_schema.db")

        # copy the db to a new file
        dst_db_path = current_path.joinpath("test_databases/test_original_schema_copy.db")
        shutil.copy(str(db_path), str(dst_db_path))
        self.assertTrue(dst_db_path.is_file())

        # open the new db
        test_database = db.Database(db_path=dst_db_path.as_posix())

        self.assertFalse(tables.DbReplayTable.has_extra_info(test_database._engine))
        self.assertTrue(test_database.is_update_required())

        # make sure we raise the exception on newer columns
        self.assertRaises(sqlalchemy.exc.OperationalError, test_database.record_setup, description="test description", name="test name", request={"test":"this is a test request"})
        test_database._session.rollback()
        test_database.perform_updates()

        # retry adding rows
        master_table = test_database.record_setup(description="test description", name="test name", request={"test":"this is a test request"})
        test_database.record_message(master_table, "test", request={"test":"this is a test request"})
        # make sure the new column exists
        self.assertTrue(tables.DbReplayTable.has_extra_info(test_database._engine))

        test_database.delete_db()


    def test_strat_table(self):
        test_strat = {"strat":"this is nothing like a strat"}
        master_table = self._database.record_setup(description="test description", name="test name", request={"test":"this is a test request"})
        # record a strat
        self._database.record_strat(master_table, strat=test_strat)
        experiment_id = master_table.experiment_id
        strat = self._database.get_strat_for(experiment_id)
        self.assertEqual(test_strat, strat)

    def test_config_table(self):
        test_config = {"config":"this is nothing like a config but it works."}
        master_table = self._database.record_setup(description="test description", name="test name", request={"test":"this is a test request"})
        # record a strat
        self._database.record_config(master_table, config=test_config)

        experiment_id = master_table.experiment_id

        config = self._database.get_config_for(experiment_id)

        self.assertEqual(test_config, config)
