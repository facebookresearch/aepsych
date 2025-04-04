#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest
from pathlib import Path
from types import ModuleType

from aepsych.config import Config
from aepsych.extensions import ExtensionManager


class ExtensionsTest(unittest.TestCase):
    extension_path = Path(__file__).parent.parent / "extensions_example"

    def tearDown(self):
        if hasattr(self, "exts"):
            self.exts.unload()

    def test_load_extensions(self):
        self.exts = ExtensionManager(
            extensions=[self.extension_path / "new_objects.py"]
        )
        self.exts.load()

        self.assertTrue(len(self.exts.loaded_modules) == 1)

        module = self.exts.loaded_modules["new_objects"]
        self.assertIsInstance(module, ModuleType)
        self.assertTrue(hasattr(module, "OnesGenerator"))
        self.assertIn("new_objects", sys.modules)

    def test_raise_missing_extension(self):
        self.exts = ExtensionManager(
            extensions=[
                self.extension_path / "new_objects.py",
                self.extension_path / "missing_extension.py",
            ]
        )
        with self.assertRaises(FileNotFoundError):
            self.exts.load()

    def test_double_load(self):
        self.exts = ExtensionManager(
            extensions=[self.extension_path / "new_objects.py"]
        )
        self.exts.load()

        with self.assertLogs() as log:
            self.exts.load()

        self.assertIn("already loaded", log[-1][0])

    def test_unload(self):
        self.exts = ExtensionManager(
            extensions=[self.extension_path / "new_objects.py"]
        )
        self.exts.load()
        self.exts.unload("new_objects")

        self.assertEqual(len(self.exts.loaded_modules), 0)
        self.assertNotIn("new_objects", sys.modules)
        self.assertNotIn("OnesGenerator", Config.registered_names)
