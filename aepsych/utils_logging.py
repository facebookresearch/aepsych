#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import logging.config
import os
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

    aepsych_mode = os.environ.get("aepsych_mode", "")
    if aepsych_mode == "test":
        logging_config["loggers"][""] = {}
        logger.setLevel(logging.CRITICAL)

    logging.config.dictConfig(logging_config)

    return logger
