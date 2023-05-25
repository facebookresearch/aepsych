#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)


def handle_can_model(server, request):
    # Check if the strategy has finished initialization; i.e.,
    # if it has a model and data to fit (strat.can_fit)
    logger.debug("got can_model message!")
    if not server.is_performing_replay:
        server.db.record_message(
            master_table=server._db_master_record, type="can_model", request=request
        )
    return {"can_model": server.strat.can_fit}
