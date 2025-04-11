#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import logging.config
import os
import re
import warnings
from typing import Any

logger = logging.getLogger()


class ColorFormatter(logging.Formatter):
    grey = "\x1b[0;30m"
    white = "\x1b[0;37m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    my_format = "%(asctime)-15s [%(levelname)-7s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + my_format + reset,
        logging.INFO: white + my_format + reset,
        logging.WARNING: yellow + my_format + reset,
        logging.ERROR: red + my_format + reset,
        logging.CRITICAL: bold_red + my_format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def getLogger(level=logging.INFO, log_path: str = "logs") -> logging.Logger:
    """Get a logger with the specified level and log path.

    Args:
        level: logging level. Default is logging.INFO.
        log_path (str): path to save the log file. Default is "logs".

    Returns:
        logger: a logger object.
    """
    os.makedirs(log_path, exist_ok=True)

    logging_config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {"standard": {"()": ColorFormatter}},
        "handlers": {
            "default": {
                "level": level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": logging.DEBUG,
                "filename": f"{log_path}/aepsych_server.log",
                "formatter": "standard",
            },
        },
        "loggers": {
            "": {"handlers": ["default", "file"], "level": level, "propagate": False},
        },
    }

    aepsych_mode = os.environ.get("AEPSYCH_MODE", "")
    ci = os.environ.get("CI", "false")
    if aepsych_mode == "test" or ci == "true":
        _set_test_warning_filters()

        logging_config["filters"] = {"test_filter": {"()": TestFilter}}
        logging_config["handlers"]["default"]["filters"] = ["test_filter"]
        del logging_config["handlers"]["file"]
        logging_config["loggers"][""]["handlers"] = ["default"]
        logging_config["loggers"][""]["level"] = logging.WARNING

    logging.config.dictConfig(logging_config)

    return logger


_IGNORED_WARNINGS = [
    r"A not p\.d\., added jitter of .* to the diagonal",
    r"Optimization failed within `scipy\.optimize\.minimize` with status \d+ and message ABNORMAL_TERMINATION_IN_LNSRCH",
    r"`scipy_minimize` terminated with status \d+, displaying original message from `scipy\.optimize\.minimize`: ABNORMAL_TERMINATION_IN_LNSRCH",
    r"Optimization failed in `gen_candidates_scipy` with the following warning\(s\):",
    r"`scipy_minimize` terminated with status OptimizationStatus\.FAILURE, displaying original message from `scipy\.optimize\.minimize`: ABNORMAL_TERMINATION_IN_LNSRCH",
    r"Optimization failed on the second try, after generating a new set of initial conditions\.",
    r"Matplotlib is building the font cache; this may take a moment\.",
    r"Skipping device Apple Paravirtual device that does not support Metal 2\.0",
]


def _set_test_warning_filters():
    # Used to ignore warnings unhelpful warnings during testing or CI
    aepsych_mode = os.environ.get("AEPSYCH_MODE", "")
    ci = os.environ.get("CI", "false")
    if aepsych_mode == "test" or ci == "true":
        warning = "|".join(_IGNORED_WARNINGS)
        warnings.filterwarnings("ignore", message=warning)

        compiled_warnings = re.compile(warning)

        def raise_warnings(message, category, filename, lineno, file=None, line=None):
            # Makes warnings that are printed case exceptions, mimics real showwarning
            msg = warnings.WarningMessage(
                message, category, filename, lineno, file, line
            )
            # Double check if this warning is ignored
            if compiled_warnings.search(str(msg)) is not None:
                return

            warnings._showwarnmsg_impl(msg)
            raise message

        warnings.showwarning = raise_warnings
        # warnings.simplefilter("ignore")


class TestFilter(logging.Filter):
    patterns = [
        # TODO: Completely remove the need of the filter by fixing this warning
        r"^Parameter-specific bounds are incomplete, falling back to ub/lb in \[common\]",
    ]

    def __init__(self, name=""):
        super().__init__(name)

        # Compile regex
        self.re = re.compile("|".join(self.patterns))

    def filter(self, record):
        """Given a record, return True if we fail to match (thus log it)."""
        return self.re.search(record.msg) is None
