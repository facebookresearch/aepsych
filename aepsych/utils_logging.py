#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import logging.config
import os

logger = logging.getLogger()


def getLogger(level: int = logging.INFO, log_path: str = "logs") -> logging.Logger:
    """Get a logger with the specified level and log path.
    
    Args:
        Level (int, optional): The logging level. Defaults to logging.INFO.
        log_path (str, optional): The path to save the logs. Defaults to "logs".
        
    Returns:
        logging.Logger: The logger object.
    """ 
    my_format = "%(asctime)-15s [%(levelname)-7s] %(message)s"
    os.makedirs(log_path, exist_ok=True)

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
                "filename": f"{log_path}/bayes_opt_server.log",
                "formatter": "standard",
            },
        },
        "loggers": {
            "": {"handlers": ["default", "file"], "level": level, "propagate": False},
        },
    }

    logging.config.dictConfig(logging_config)
    return logger
