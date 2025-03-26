#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Literal, TypedDict

import aepsych.utils_logging as utils_logging
import numpy as np
import torch

logger = utils_logging.getLogger(logging.INFO)

QueryResponse = TypedDict(
    "QueryResponse",
    {
        "query_type": str,
        "probability_space": bool,
        "constraints": dict[int, float],
        "x": dict[str, np.ndarray],
        "y": np.ndarray,
    },
)


def handle_query(server, request: dict[str, Any]) -> QueryResponse | None:
    """Queries the underlying model given a specific query request.

    Args:
        server (AEPsychServer): AEPsych server responding to the message.
        request (dict[str, Any]): A dictionary from the request message.

    Returns:
        QueryResponse, optional: None if server is skipping computations for a replay.
            Otherwise, returns a dictionary with these entries:
            - "query_response": string, the query response.
            - "probability_space": boolean, whether to query in the probability space
                or not.
            - "constraints": dictionary, the equality constraint for parameters
                where the keys are the parameter index and the values are the point
                where the paramter should be constrained to.
            - "x": dictionary, the parameter configuration dictionary for the query.
            - "y": np.ndarray, the y from the query.
    """
    logger.debug("got query message!")
    if not server.is_performing_replay:
        server.db.record_message(
            master_table=server._db_master_record, type="query", request=request
        )
    response = query(server, **request["message"])
    return response


def query(
    server,
    query_type: Literal["max", "min", "prediction", "inverse"] = "max",
    probability_space: bool = False,
    x: dict[str, Any] | None = None,
    y: float | torch.Tensor | None = None,
    constraints: dict[int, float] | None = None,
    **kwargs,
) -> QueryResponse | None:
    """Queries the underlying model for a specific query.

    Args:
        query_type (Literal["max", "min", "prediction", "inverse"]): What type of query
            to make. Defaults to "max".
        probability_space (bool): Whether the y in the query is in probability space.
            Defaults to False.
        x (dict[str, Any], optional): A parameter configuration dictionary representing
            one or more point for a prediction query.
        y (float | torch.Tensor, optional): The expected y for a inverse query.
        constraints (dict[int, float], optional): The constraints to impose on the
            query where each key is the parameter index and the value is the parameter
            value to apply the equality constraint at.
        **kwargs: Additional kwargs to pass to the query function.

    Returns:
        QueryResponse, optional: None if server is skipping computations for a replay.
            Otherwise, returns a dictionary with these entries:
            - "query_response": string, the query response.
            - "probability_space": boolean, whether to query in the probability space
                or not.
            - "constraints": dictionary, the equality constraint for parameters
                where the keys are the parameter index and the values are the point
                where the paramter should be constrained to.
            - "x": dictionary, the parameter configuration dictionary for the query.
            - "y": np.ndarray, the y from the query.
    """
    if server.skip_computations:
        return None

    constraints = constraints or {}
    response: QueryResponse = {
        "query_type": query_type,
        "probability_space": probability_space,
        "constraints": constraints,
        "x": {"placeholder": np.empty(0)},
        "y": np.empty(0),
    }
    if query_type == "max":
        fmax, fmax_loc = server.strat.get_max(constraints, probability_space, **kwargs)
        response["y"] = fmax
        response["x"] = server._tensor_to_config(fmax_loc)
    elif query_type == "min":
        fmin, fmin_loc = server.strat.get_min(constraints, probability_space, **kwargs)
        response["y"] = fmin
        response["x"] = server._tensor_to_config(fmin_loc)
    elif query_type == "prediction":
        # returns the model value at x
        if x is None:  # TODO: ensure if x is between lb and ub
            raise RuntimeError("Cannot query model at location = None!")

        mean, _var = server.strat.predict(
            server._config_to_tensor(x),
            probability_space=probability_space,
        )
        response["x"] = x
        y = mean.item() if isinstance(mean, torch.Tensor) else mean[0]
        response["y"] = np.array(y)  # mean.item()

    elif query_type == "inverse":
        nearest_y, nearest_loc = server.strat.inv_query(
            y, constraints, probability_space=probability_space, **kwargs
        )
        response["y"] = np.array(nearest_y)
        response["x"] = server._tensor_to_config(nearest_loc)
    else:
        raise RuntimeError("unknown query type!")
    # ensure all x values are arrays
    response["x"] = {
        k: np.array([v]) if np.array(v).ndim == 0 else v
        for k, v in response["x"].items()
    }

    return response
