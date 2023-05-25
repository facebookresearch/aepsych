#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)


def handle_exit(server, request):
    # Make local server write strats into DB and close the connection
    termination_type = "Normal termination"
    logger.info("Got termination message!")
    server.write_strats(termination_type)
    if not server.is_using_thrift:
        server.exit_server_loop = True

    # If using thrift, it will add 'Terminate' to the queue and pass it to thrift server level
    return "Terminate"
