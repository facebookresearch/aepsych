#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import aepsych.server as server
import aepsych.utils_logging as utils_logging
from aepsych.config import Config

def init_server(db_path):
    # setup logger
    server.logger = utils_logging.getLogger(logging.DEBUG, "logs")
    # random port
    socket = server.sockets.PySocket(port=0)
    # random datebase path name without dashes
    return server.AEPsychServer(socket=socket, database_path=db_path)

def generate_single_stimuli_db():
    # Read config from .ini file
    with open('tests/configs/singleStimuli_singleOutcome.ini', 'r') as f:
        dummy_simple_config = f.read()

    setup_request = {
        "type": "setup",
        "version": "0.01",
        "message": {"config_str": dummy_simple_config},
    }
    ask_request = {"type": "ask", "message": ""}
    tell_request = {
        "type": "tell",
        "message": {"config": {"x1": [0.0], "x2":[0.0]}, "outcome": 1},
        "extra_info": {},
    }

    s = init_server(db_path = "single_stimuli.db")
    s.versioned_handler(setup_request)

    x1 = [0.1, 0.2, 0.3, 1, 2, 3, 4]
    x2 = [4, 0.1, 3, 0.2, 2, 1, 0.3]
    outcomes = [1, -1, 0.1, 0, -0.1, 0, 0]

    i = 0
    while not s.strat.finished:
        s.unversioned_handler(ask_request)
        tell_request["message"]["config"]["x1"] = [x1[i]]
        tell_request["message"]["config"]["x2"] = [x2[i]]
        tell_request["message"]["outcome"] = outcomes[i]
        tell_request["extra_info"]["e1"] = 1
        tell_request["extra_info"]["e2"] = 2
        i = i + 1
        s.unversioned_handler(tell_request)

def generate_multi_stimuli_db():
    # Read config from .ini file
    with open('tests/configs/multiStimuli_multiOutcome.ini', 'r') as f:
        dummy_pairwise_config = f.read()

    setup_request = {
        "type": "setup",
        "version": "0.01",
        "message": {"config_str": dummy_pairwise_config},
    }
    ask_request = {"type": "ask", "message": ""}
    tell_request = {
        "type": "tell",
        "message": {"config": {"par1": [0.0], "par2":[0.0]}, "outcome": [0, 0]},
        "extra_info": {},
    }

    s = init_server(db_path = "multi_stimuli.db")
    s.versioned_handler(setup_request)

    par1 = [[0.1, 0.2], [0.3, 1], [2, 3], [4, 0.1], [0.2, 2], [1, 0.3], [0.3, 0.1]]
    par2 = [[4, 0.1], [3, 0.2], [2, 1], [0.3, 0.2], [2, 0.3], [1, 0.1], [0.3, 4]]
    outcomes = [[1, 0], [-1, 0], [0.1, 0], [0, 0], [-0.1, 0], [0, 0], [0, 0]]

    i = 0
    while not s.strat.finished:
        s.unversioned_handler(ask_request)
        tell_request["message"]["config"]["par1"] = par1[i]
        tell_request["message"]["config"]["par2"] = par2[i]
        tell_request["message"]["outcome"] = outcomes[i]
        tell_request["extra_info"]["e1"] = 1
        tell_request["extra_info"]["e2"] = 2
        i = i + 1
        s.unversioned_handler(tell_request)

if __name__ == "__main__":
    generate_single_stimuli_db()
    generate_multi_stimuli_db()
