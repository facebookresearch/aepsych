#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, TypedDict

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)

CanModelResponse = TypedDict("CanModelResponse", {"can_model": bool})


def handle_can_model(server, request: Dict[str, Any]) -> CanModelResponse:
    """Check if the strategy has finished initialization and a model is ready.

    Args:
        server (AEPsychServer): AEPsych server responding to the message.
        request (Dict[str, Any]): A dictionary from the request message.

    Returns:
        CanModelResponse: A dictionary with one entry.
            - "can_model": boolean, whether the current strategy has a ready model
                (based on if there is a model and if it has data to fit on).
    """
    logger.debug("got can_model message!")
    if not server.is_performing_replay:
        server.db.record_message(
            master_table=server._db_master_record, type="can_model", request=request
        )
    return {"can_model": server.strat.can_fit}
