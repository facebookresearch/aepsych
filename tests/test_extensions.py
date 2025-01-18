#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from types import ModuleType

from aepsych.extensions import ExtensionManager


class ExtensionsTest(unittest.TestCase):
    extension_path = Path(__file__).parent / "extensions"

    def test_load_extensions(self):
        exts = ExtensionManager(files=[self.extension_path / "register_object.py"])
        exts.load()

        self.assertTrue(len(exts.loaded_modules) == 1)

        module = exts.loaded_modules["register_object"]
        self.assertIsInstance(module, ModuleType)
        self.assertTrue(hasattr(module, "OnesGenerator"))

    def test_raise_missing_extension(self):
        exts = ExtensionManager(
            files=[
                self.extension_path / "register_object.py",
                self.extension_path / "missing_extension.py",
            ]
        )
        with self.assertRaises(FileNotFoundError):
            exts.load()

    def test_double_load(self):
        exts = ExtensionManager(files=[self.extension_path / "register_object.py"])
        exts.load()

        with self.assertWarns(UserWarning):
            exts.load()
