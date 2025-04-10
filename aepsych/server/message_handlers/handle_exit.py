#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, TypedDict

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)

ExitResponse = TypedDict("ExitResponse", {"termination_type": str, "success": bool})


def handle_exit(server, request: dict[str, Any]) -> ExitResponse:
    """Make local server write strats into DB and close the connection.

    Args:
        server (AEPsychServer): AEPsych server responding to the message.
        request (dict[str, Any]): A dictionary from the request message. Currently
            ignored.

    Returns:
        ExitResponse: A dictionary with two entries:
            - "config": dictionary with config (keys are strings, values are floats).
                Currently always "Terminate" if this function succeeds.
            - "is_finished": boolean, true if the strat is finished. Currently always
                true if this function succeeds.
    """
    termination_type = "Normal termination"
    logger.info("got termination message!")
    server.write_strats(termination_type)
    server.exit_server_loop = True

    return {"termination_type": "Terminate", "success": True}
