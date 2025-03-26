#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, TypedDict

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)

ResumeResponse = TypedDict("ResumeResponse", {"strat_id": int})


def handle_resume(server, request: dict[str, Any]) -> ResumeResponse:
    """Resume a specific strategy given its ID.

    Args:
        server (AEPsychServer): AEPsych server responding to the message.
        request (dict[str, Any]): A dictionary from the request message, must include
            the "strat_id" key in its message.

    Returns:
        ResumeResponse: A dictionary with one entry
            - "strat_id": integer, the stategy ID that was resumed.
    """
    logger.debug("got resume message!")
    strat_id = int(request["message"]["strat_id"])
    server.strat_id = strat_id
    if not server.is_performing_replay:
        server.db.record_message(
            master_table=server._db_master_record, type="resume", request=request
        )
    return {"strat_id": server.strat_id}
