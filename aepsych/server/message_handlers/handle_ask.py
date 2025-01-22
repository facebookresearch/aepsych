#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Mapping

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)


def handle_ask(server, request):
    """Returns dictionary with two entries:
    "config" -- dictionary with config (keys are strings, values are floats)
    "is_finished" -- bool, true if the strat is finished
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
        lengths = {len(val) for val in params.values()}

    new_config = {
        "config": params,
        "is_finished": server.strat.finished,
        "num_points": lengths.pop(),
    }
    if not server.is_performing_replay:
        server.db.record_message(
            master_table=server._db_master_record, type="ask", request=request
        )
    return new_config


def ask(server, num_points=1, **kwargs):
    """get the next point to query from the model
    Returns:
        dict -- new config dict (keys are strings, values are floats)
    """
    if server.skip_computations:
        # Fakes points being generated and tracked by strategy to end strategies
        server.strat._strat._count += num_points
        if server.strat._strat.finished:
            server.strat._make_next_strat()
        return None

    # The fixed_pars kwargs name is purposefully differend to the fixed_features
    # expected by botorch's optimize acqf to avoid doubling up ever while allowing other
    # kwargs to pass through
    if "fixed_pars" in kwargs:
        fixed_pars = kwargs.pop("fixed_pars")
        kwargs["fixed_features"] = server._fixed_to_idx(fixed_pars)

    next_x = server.strat.gen(num_points=num_points, **kwargs)
    return server._tensor_to_config(next_x)
