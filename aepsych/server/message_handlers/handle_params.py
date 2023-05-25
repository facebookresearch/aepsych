#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)


def handle_params(server, request):
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
