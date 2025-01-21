import importlib.util
import logging
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

from aepsych.config import ConfigurableMixin
from aepsych.utils_logging import getLogger

logger = getLogger()


class ExtensionManager(ConfigurableMixin):
    """Class to manage extensions for a config. Loads each extension (which can just be
    simple Python scripts) as a module. In the future, extensions may have specific
    requirements that the extension manager will expect (e.g., unload)."""

    def __init__(self, files: List[str]) -> None:
        """Initialize the ExtensionManager. Each extension is represented by a path to
        the script.

        Args:
            files (List[str]): List of file paths of extension scripts.
        """
        self.ext_files = {Path(path).stem: Path(path) for path in files}
        self.loaded_modules: Dict[str, ModuleType] = {}

    def load(self) -> None:
        """Load all extensions in ExtensionManager"""
        for ext_name, ext_path in self.ext_files.items():
            _ = self._import_extension(ext_name)
            logging.info(f"Extension at {ext_path} loaded as {ext_name}.")

    def _import_extension(self, name: str) -> ModuleType:
        # Given an extension name, import the module, returning it
        if name in self.loaded_modules:
            warnings.warn(f"The extension '{name}' is already loaded.", UserWarning)
            return self.loaded_modules[name]

        # Creates a module from a file
        spec = importlib.util.spec_from_file_location(name, self.ext_files[name])
        module = importlib.util.module_from_spec(spec)  # type: ignore
        sys.modules[name] = module
        spec.loader.exec_module(module)  # type: ignore

        self.loaded_modules[name] = module

        return module

    @classmethod
    def get_config_options(
        cls,
        config,
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Find the config options for the extension manager.

        Args:
            config (Config): Config to look for options in.
            name (str, optional): Unused, kept for API conformity.
            options (Dict[str, Any], optional): Existing options, any key in options
                will be ignored from the config.

        Return:
            Dict[str, Any]: A dictionary of options to initialize ExtensionManager.
        """
        if options is None:
            options = {}

        if "files" in options:  # Already have what we need
            return options
        elif "extensions" in config["common"]:
            options["files"] = config.getlist("common", "extensions", element_type=str)
        else:
            options["files"] = {}

        return options
