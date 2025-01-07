#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from pathlib import Path


class CLITestCase(unittest.TestCase):
    def test_summarize_cli(self):
        current_path = Path(os.path.abspath(__file__)).parent
        db_path = current_path
        db_path = db_path.joinpath("test_databases/1000_outcome.db")
        exit_status = os.system(f"aepsych_database --db {db_path} --summarize")
        self.assertEqual(exit_status, 0)

    def test_to_csv_cli(self):
        current_path = Path(os.path.abspath(__file__)).parent.parent
        db_path = current_path.joinpath("test_databases/1000_outcome.db")
        csv_path = current_path.joinpath("test_csv.csv")
        exit_status = os.system(f"aepsych_database --db {db_path} --tocsv {csv_path}")
        self.assertEqual(exit_status, 0)

        self.assertTrue(os.path.exists(csv_path))
        os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()
