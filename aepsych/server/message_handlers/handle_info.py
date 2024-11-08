#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)


def handle_info(server, request: Dict[str, Any]) -> Dict[str, Any]:
    """Handles info message from the client.

    Args:
        request (Dict[str, Any]): The info message from the client

    Returns:
        Dict[str, Any]: Returns dictionary containing the current state of the experiment
    """
    logger.debug("got info message!")

    ret_val = info(server)

    return ret_val


def info(server) -> Dict[str, Any]:
    """Returns details about the current state of the server and experiments

    Returns:
        Dict: Dict containing server and experiment details
    """
    current_strat_model = (
        server.config.get(server.strat.name, "model", fallback="model not set")
        if server.config and ("model" in server.config.get_section(server.strat.name))
        else "model not set"
    )
    current_strat_acqf = (
        server.config.get(server.strat.name, "acqf", fallback="acqf not set")
        if server.config and ("acqf" in server.config.get_section(server.strat.name))
        else "acqf not set"
    )

    response = {
        "db_name": server.db._db_name,
        "exp_id": server._db_master_record.experiment_id,
        "strat_count": server.n_strats,
        "all_strat_names": server.strat_names,
        "current_strat_index": server.strat_id,
        "current_strat_name": server.strat.name,
        "current_strat_data_pts": (
            server.strat.x.shape[0] if server.strat.x is not None else 0
        ),
        "current_strat_model": current_strat_model,
        "current_strat_acqf": current_strat_acqf,
        "current_strat_finished": server.strat.finished,
    }

    logger.debug(f"Current state of server: {response}")
    return response
