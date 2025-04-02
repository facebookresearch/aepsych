#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Any

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)


def handle_get_config(server, request: dict[str, Any]) -> dict[str, Any]:
    """Return the current experiment config. The message can specify a section and
    property to get the specific property from a specific section but otherwise the
    entire config is returned.

    Args:
        server (AEPsychServer): AEPsych server responding to the message.
        request (dict[str, Any]): A dictionary from the request message. Can include
            both a section and property key to get a specific property instead of the
            whole experiment config.

    Returns:
        dict[str, Any]: A dictionary representing the experiment config.
    """
    msg = request["message"]
    section = msg.get("section", None)
    prop = msg.get("property", None)

    # If section and property are not specified, return the whole config
    if section is None and prop is None:
        return server.config.to_dict(deduplicate=False)

    # If section and property are not both specified, raise an error
    if section is None and prop is not None:
        raise RuntimeError("Message contains a property but not a section!")
    if section is not None and prop is None:
        raise RuntimeError("Message contains a section but not a property!")

    # If both section and property are specified, return only the relevant value from the config
    return {section: {prop: server.config.to_dict(deduplicate=False)[section][prop]}}
