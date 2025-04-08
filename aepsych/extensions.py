#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

from aepsych.config import ConfigurableMixin
from aepsych.utils_logging import getLogger

logger = getLogger()


class ExtensionManager(ConfigurableMixin):
    """Class to manage extensions for a config. Loads each extension (which can just be
    simple Python scripts) as a module. In the future, extensions may have specific
    requirements that the extension manager will expect (e.g., unload)."""

    def __init__(self, extensions: list[str] | None = None) -> None:
        """Initialize the ExtensionManager. Each extension is represented by a path to
        the script.

        Args:
            files (list[str]): List of file paths of extension scripts.
        """
        if extensions is not None:
            self.ext_files = {Path(path).stem: Path(path) for path in extensions}
        else:
            self.ext_files = {}
        self.loaded_modules: dict[str, ModuleType] = {}

    def load(self) -> None:
        """Load all extensions in ExtensionManager"""
        for ext_name, ext_path in self.ext_files.items():
            _ = self._import_extension(ext_name)
            logging.info(f"Extension at {ext_path} loaded as {ext_name}.")

    def _import_extension(self, name: str) -> ModuleType:
        # Given an extension name, import the module, returning it
        if name in self.loaded_modules:
            logger.warning(f"The extension '{name}' is already loaded.")
            return self.loaded_modules[name]

        # Creates a module from a file
        spec = importlib.util.spec_from_file_location(name, self.ext_files[name])
        module = importlib.util.module_from_spec(spec)  # type: ignore
        sys.modules[name] = module
        spec.loader.exec_module(module)  # type: ignore

        self.loaded_modules[name] = module

        return module

    def unload(self, extensions: str | list[str] | None = None):
        """Unload extensions. Removes the extension module from the module cache. If the
        module has the _unload function defined, it will also run that before
        unloading.

        Args:
            extensions (str | list[str], optional): Extension to be unloaded. If not
                set, we will attempt to unload all extensions with the unload function
                defined.
        """

        if isinstance(extensions, str):
            extensions = [extensions]
        elif extensions is None:
            extensions = list(self.loaded_modules.keys())

        for extension in extensions:
            if hasattr(self.loaded_modules[extension], "_unload"):
                self.loaded_modules[extension]._unload()

            del self.loaded_modules[extension]
            del sys.modules[extension]
