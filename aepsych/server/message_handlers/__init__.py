#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .handle_ask import handle_ask
from .handle_can_model import handle_can_model
from .handle_exit import handle_exit
from .handle_finish_strategy import handle_finish_strategy
from .handle_get_config import handle_get_config
from .handle_info import handle_info
from .handle_params import handle_params
from .handle_query import handle_query
from .handle_resume import handle_resume
from .handle_setup import handle_setup
from .handle_tell import handle_tell

MESSAGE_MAP = {
    "setup": handle_setup,
    "ask": handle_ask,
    "tell": handle_tell,
    "query": handle_query,
    "parameters": handle_params,
    "can_model": handle_can_model,
    "exit": handle_exit,
    "get_config": handle_get_config,
    "finish_strategy": handle_finish_strategy,
    "info": handle_info,
    "resume": handle_resume,
}
