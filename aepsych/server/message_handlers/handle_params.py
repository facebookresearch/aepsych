#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)


def handle_params(server, request: dict[str, Any]) -> dict[str, list[float]]:
    """Returns a dictionary about each parameter in the current strategy.

    Args:
        server (AEPsychServer): AEPsych server responding to the message.
        request (dict[str, Any]): A dictionary from the request message. Currently
            ignored expect to record the message in the replay.

    Returns:
        dict[str, list[float]]: A dictionary where every key is a parameter and the
            value is a list of floats representing the lower bound and the upper bound.
    """
    logger.debug("got parameters message!")
    if not server.is_performing_replay:
        server.db.record_message(
            master_table=server._db_master_record, type="parameters", request=request
        )
    config_setup = {
        server.parnames[i]: [server.strat.lb[i].item(), server.strat.ub[i].item()]
        for i in range(len(server.parnames))
    }
    return config_setup
