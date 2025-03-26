#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, TypedDict

FinishStrategyResponse = TypedDict(
    "FinishStrategyResponse", {"finished_strategy": str, "finished_strat_idx": int}
)


def handle_finish_strategy(server, request: dict[str, Any]) -> FinishStrategyResponse:
    """Finish the current strategy and return a dictionary describing the finished
    strategy.

    Args:
        server (AEPsychServer): AEPsych server responding to the message.
        request (dict[str, Any]): A dictionary from the request message. Currently
            ignored.

    Returns:
        FinishStrategyResponse: A dictionary with these entries
            - "finished_strategy": str, name of the finished strategy.
            - "finished_strat_idx": int, the id of the strategy.
    """

    server.strat.finish()
    return {
        "finished_strategy": server.strat.name,
        "finished_strat_idx": server.strat._strat_idx,
    }
