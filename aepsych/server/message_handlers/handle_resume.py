#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)


def handle_resume(server, request):
    logger.debug("got resume message!")
    strat_id = int(request["message"]["strat_id"])
    server.strat_id = strat_id
    if not server.is_performing_replay:
        server.db.record_message(
            master_table=server._db_master_record, type="resume", request=request
        )
    return server.strat_id
