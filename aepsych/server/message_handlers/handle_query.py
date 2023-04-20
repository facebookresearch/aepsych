import logging
import numpy as np
import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.INFO)


def handle_query(server, request):
    logger.debug("got query message!")
    if not server.is_performing_replay:
        server.db.record_message(
            master_table=server._db_master_record, type="query", request=request
        )
    response = query(server, **request["message"])
    return response


def query(
    server,
    query_type="max",
    probability_space=False,
    x=None,
    y=None,
    constraints=None,
):
    if server.skip_computations:
        return None

    constraints = constraints or {}
    response = {
        "query_type": query_type,
        "probability_space": probability_space,
        "constraints": constraints,
    }
    if query_type == "max":
        fmax, fmax_loc = server.strat.get_max(constraints)
        response["y"] = fmax.item()
        response["x"] = server._tensor_to_config(fmax_loc)
    elif query_type == "min":
        fmin, fmin_loc = server.strat.get_min(constraints)
        response["y"] = fmin.item()
        response["x"] = server._tensor_to_config(fmin_loc)
    elif query_type == "prediction":
        # returns the model value at x
        if x is None:  # TODO: ensure if x is between lb and ub
            raise RuntimeError("Cannot query model at location = None!")
        mean, _var = server.strat.predict(
            server._config_to_tensor(x).unsqueeze(axis=0),
            probability_space=probability_space,
        )
        response["x"] = x
        response["y"] = mean.item()
    elif query_type == "inverse":
        # expect constraints to be a dictionary; values are float arrays size 1 (exact) or 2 (upper/lower bnd)
        constraints = {server.parnames.index(k): v for k, v in constraints.items()}
        nearest_y, nearest_loc = server.strat.inv_query(
            y, constraints, probability_space=probability_space
        )
        response["y"] = nearest_y
        response["x"] = server._tensor_to_config(nearest_loc)
    else:
        raise RuntimeError("unknown query type!")
    # ensure all x values are arrays
    response["x"] = {
        k: np.array([v]) if np.array(v).ndim == 0 else v
        for k, v in response["x"].items()
    }
    return response
