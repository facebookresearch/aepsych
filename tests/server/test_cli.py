#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import unittest
import uuid
from pathlib import Path

from aepsych.database.db import Database
from aepsych.server.utils import parse_argument, run_database


class CLITestCase(unittest.TestCase):
    def test_summarize_cli(self):
        current_path = Path(os.path.abspath(__file__)).parent.parent
        db_path = current_path
        db_path = db_path.joinpath("test_databases/1000_outcome.db")
        args = parse_argument(["--db", str(db_path), "--summarize"])
        with unittest.mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            run_database(args)
            out = mock_stdout.getvalue()

        self.assertIn("experiment_id", out)
        self.assertIn("row", out)
        self.assertIn("column", out)

    def test_to_csv_cli(self):
        current_path = Path(os.path.abspath(__file__)).parent.parent
        db_path = current_path.joinpath("test_databases/1000_outcome.db")
        csv_path = current_path.joinpath(f"test_csv_{str(uuid.uuid4().hex)}.csv")
        args = parse_argument(["--db", str(db_path), "--tocsv", str(csv_path)])

        with self.assertLogs(level="INFO") as log:
            run_database(args)

        self.assertIn("Exported contents of ", log[-1][-1])
        self.assertIn(str(csv_path), log[-1][-1])
        self.assertTrue(os.path.exists(csv_path))
        os.remove(csv_path)

    def test_combine(self):
        """Test combining databases"""
        current_path = Path(os.path.abspath(__file__)).parent.parent
        db_path = current_path / "test_databases"
        out_path = current_path / f"combined_db_{str(uuid.uuid4().hex)}.db"
        args = parse_argument(
            [
                "--db",
                str(out_path),
                "--combine",
                str(db_path),
                "--exclude",
                str(db_path / "1000_outcome.db"),  # Takes forever
                "--exclude",
                str(db_path / "test_original_schema.db"),  # If included, will error out
            ]
        )

        with unittest.mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            with self.assertLogs(level="INFO") as log:
                run_database(args)

        self.assertIn("Found 3 dbs to combine.", mock_stdout.getvalue())
        self.assertIn(f"Combined 3 experiment sessions into {out_path}", log[-1][-1])
        self.assertTrue(out_path.exists())

        # Open the db and check that it everything
        db = Database(str(out_path))

        # Last experiment to be added has 15 messages
        records = db.get_master_records()
        self.assertEqual(len(records), 3)
        self.assertEqual(len(db.get_replay_for(records[-1].unique_id)), 15)
        self.assertIn("single_stimuli.db", records[-1].extra_metadata)

        # Look through all master records to check if we don't include excluded
        for record in records:
            self.assertNotIn("1000_outcome.db", record.extra_metadata)
            self.assertNotIn("test_original_schema.db", record.extra_metadata)

        db.delete_db()

    def test_combine_file_exists_error(self):
        """Test that combining databases raises FileExistsError when output path exists"""
        current_path = Path(os.path.abspath(__file__)).parent.parent
        db_path = current_path / "test_databases"

        # Create a file at the output path
        out_path = current_path / f"existing_db_{str(uuid.uuid4().hex)}.db"
        with open(out_path, "w") as f:
            f.write("This file already exists")

        self.assertTrue(out_path.exists())

        args = parse_argument(
            [
                "--db",
                str(out_path),
                "--combine",
                str(db_path),
            ]
        )

        # Assert that FileExistsError is raised
        with self.assertRaises(FileExistsError):
            run_database(args)

        # Clean up
        out_path.unlink()


if __name__ == "__main__":
    unittest.main()
