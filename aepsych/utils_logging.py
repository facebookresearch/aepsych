#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import logging.config
import os

logger = logging.getLogger()


def getLogger(level=logging.INFO) -> logging.Logger:
    my_format = "%(asctime)-15s [%(levelname)-7s] %(message)s"

    os.makedirs("logs", exist_ok=True)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {"standard": {"format": my_format}},
        "handlers": {
            "default": {
                "level": level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": logging.DEBUG,
                "filename": "logs/bayes_opt_server.log",
                "formatter": "standard",
            },
        },
        "loggers": {
            "": {"handlers": ["default", "file"], "level": level, "propagate": False},
        },
    }

    logging.config.dictConfig(logging_config)
    return logger
