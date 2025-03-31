#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import unittest
from pathlib import Path

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
        csv_path = current_path.joinpath("test_csv.csv")
        args = parse_argument(["--db", str(db_path), "--tocsv", str(csv_path)])

        with self.assertLogs(level="INFO") as log:
            run_database(args)

        self.assertIn("Exported contents of ", log[-1][-1])
        self.assertIn(str(csv_path), log[-1][-1])
        self.assertTrue(os.path.exists(csv_path))
        os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()
