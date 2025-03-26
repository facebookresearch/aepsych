#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, TypedDict

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)

InfoResponse = TypedDict(
    "InfoResponse",
    {
        "db_name": str,
        "exp_id": int,
        "strat_count": int,
        "all_strat_names": list[str],
        "current_strat_index": int,
        "current_strat_name": str,
        "current_strat_data_pts": int,
        "current_strat_model": str,
        "current_strat_acqf": str,
        "current_strat_finished": bool,
        "current_strat_can_fit": bool,
    },
)


def handle_info(server, request: dict[str, Any]) -> InfoResponse:
    """Returns info on the current running experiment.

    Args:
        server (AEPsychServer): AEPsych server responding to the message.
        request (dict[str, Any]): A dictionary from the request message. Currently
            ignored.

    Returns:
        InfoResponse: A dictionary with these entries
            - "db_name": string, name of the database
            - "exp_id": integer, experiment ID
            - "strat_count": integer, number of strategies in the server.
            - "all_strat_names": list of strings, list of the strategy names in the
                server.
            - "current_strat_index": integer, the index of the current strategy.
            - "current_strat_name": string, name of the current strategy.
            - "current_strat_data_pts": integer, the number of data points in the
                current strategy.
            - "current_strat_model": string, the name of the model in the current
                strategy.
            - "current_strat_acqf": string, the acquisition function of the current
                stratgy.
            - "current_strat_finished": boolean, whether the current strategy is
                finished.
            - "current_strat_can_fit": boolean, whether the current strategy can fit a
                model.
    """
    logger.debug("got info message!")

    ret_val = info(server)

    return ret_val


def info(server) -> InfoResponse:
    """Returns details about the current state of the server and experiments

    Args:
        server (AEPsychServer): AEPsych server to get info on.

    Returns:
        InfoResponse: A dictionary with these entries
            - "db_name": string, name of the database
            - "exp_id": integer, experiment ID
            - "strat_count": integer, number of strategies in the server.
            - "all_strat_names": list of strings, list of the strategy names in the
                server.
            - "current_strat_index": integer, the index of the current strategy.
            - "current_strat_name": string, name of the current strategy.
            - "current_strat_data_pts": integer, the number of data points in the
                current strategy.
            - "current_strat_model": string, the name of the model in the current
                strategy.
            - "current_strat_acqf": string, the acquisition function of the current
                stratgy.
            - "current_strat_finished": boolean, whether the current strategy is
                finished.
            - "current_strat_can_fit": boolean, whether the current strategy can fit a
                model.
    """
    current_strat_model = (
        server.config.get(server.strat.name, "model", fallback="model not set")
        if server.config and ("model" in server.config.get_section(server.strat.name))
        else "model not set"
    )
    try:
        current_strat_acqf = server.strat.generator.acqf.__name__
    except AttributeError:
        current_strat_acqf = "acqf not set"

    response: InfoResponse = {
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
        "current_strat_can_fit": server.strat.can_fit,
    }

    logger.debug(f"Current state of server: {response}")
    return response
