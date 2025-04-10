#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Mapping
from typing import Any, TypedDict

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)

AskResponse = TypedDict(
    "AskResponse",
    {"config": dict[str, Any] | None, "is_finished": bool, "num_points": int},
)


def handle_ask(server, request: dict[str, Any]) -> AskResponse:
    """Requests a point to be generated and return a dictionary with two entries
    representing the parameter configuration and whether or not the strategy is
    finished.

    Args:
        server (AEPsychServer): AEPsych server responding to the message.
        request (dict[str, Any]): A dictionary from the request message.

    Returns:
        AskResponse: A dictionary with three entries
            - "config": dictionary with config (keys are strings, values are floats), None
                if skipping computations during replay.
            - "is_finished": boolean, true if the strat is finished
            - "num_points": integer, number of points returned.
    """
    logger.debug("got ask message!")
    if server._pregen_asks:
        params = server._pregen_asks.pop()
    else:
        # Some clients may still send "message" as an empty string, so we need to check if its a dict or not.
        msg = request["message"]
        if isinstance(msg, Mapping):
            params = ask(server, **msg)
        else:
            params = ask(server)

    if params is None:  # For replay
        lengths = [0]
    else:
        lengths = [len(val) for val in params.values()]

    new_config: AskResponse = {
        "config": params,
        "is_finished": server.strat.finished,
        "num_points": lengths.pop(),
    }
    if not server.is_performing_replay:
        server.db.record_message(
            master_table=server._db_master_record, type="ask", request=request
        )
    return new_config


def ask(server, num_points: int = 1, **kwargs) -> dict[str, Any] | None:
    """Returns points from a generator.

    Returns:
        dict[str, Any], optional: New parameter config dict (keys are strings, values
            are floats). If the server is skipping computations, just return None.
    """
    if server.skip_computations:
        # Fakes points being generated and tracked by strategy to end strategies
        if server.strat._strat.finished:
            server.strat._make_next_strat()
        server.strat._strat._count += num_points
        return None

    # The fixed_pars kwargs name is purposefully different to the fixed_features
    # expected by botorch's optimize acqf to avoid doubling up ever while allowing other
    # kwargs to pass through
    if "fixed_pars" in kwargs:
        fixed_pars = kwargs.pop("fixed_pars")
        kwargs["fixed_features"] = server._fixed_to_idx(fixed_pars)

    next_x = server.strat.gen(num_points=num_points, **kwargs)
    return server._tensor_to_config(next_x)
