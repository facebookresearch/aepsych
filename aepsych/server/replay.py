#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings

import aepsych.utils_logging as utils_logging
import pandas as pd
from aepsych.server.message_handlers.handle_tell import flatten_tell_record

logger = utils_logging.getLogger(logging.INFO)


def replay(
    server,
    id_to_replay=None,
    skip_computations=False,
    *,
    uuid_to_replay=None,
):
    """
    Run a replay against the server. The unique ID will be looked up in the database.
    if skip_computations is true, skip all the asks and queries, which should make the replay much faster.
    """
    if uuid_to_replay is not None:
        warnings.warn(
            "replay arg 'uuid_to_replay` is not actually a uuid, just the unique ID in the DB of the specific run of an experiment. This argument will change soon.",
            DeprecationWarning,
            stacklevel=2,
        )
        id_to_replay = uuid_to_replay

    if id_to_replay is None:
        raise RuntimeError("unique ID is a required parameter to perform a replay")

    if server.db is None:
        raise RuntimeError("A database is required to perform a replay")

    if skip_computations is True:
        logger.info(
            "skip_computations=True, make sure to refit the final strat before doing anything!"
        )

    master_record = server.db.get_master_record(id_to_replay)

    # We're going to assume that a config message will be sent in this replay such that the server.strat_id matches this record
    server._db_master_record = master_record

    if master_record is None:
        raise RuntimeError(
            f"The unique ID {id_to_replay} isn't in the database. Unable to perform replay."
        )

    # this prevents writing back to the DB and creating a circular firing squad
    server.is_performing_replay = True
    server.skip_computations = skip_computations

    for result in master_record.children_replay:
        request = result.message_contents
        logger.debug(f"replay - type = {result.message_type} request = {request}")
        server.handle_request(request)

    server.is_performing_replay = False
    server.skip_computations = False


def get_strats_from_replay(
    server,
    id_of_replay=None,
    force_replay=False,
    *,
    uuid_of_replay=None,
):
    if uuid_of_replay is not None:
        warnings.warn(
            "replay arg 'uuid_of_replay` is not actually a uuid, just the unique ID in the DB of the specific run of an experiment. This argument will change soon.",
            DeprecationWarning,
            stacklevel=2,
        )
        id_of_replay = uuid_of_replay
    if id_of_replay is None:
        records = server.db.get_master_records()
        if len(records) > 0:
            id_of_replay = records[-1].unique_id
        else:
            raise RuntimeError("Server has no experiment records!")

    if force_replay:
        warnings.warn(
            "Force-replaying to get non-final strats is deprecated after the ability"
            + " to save all strats was added, and will eventually be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        replay(server, id_of_replay, skip_computations=True)
        for strat in server._strats:
            if strat.has_model:
                strat.model.fit(strat.x, strat.y)
        return server._strats
    else:
        return server.db.get_strats_for(id_of_replay)


def get_strat_from_replay(
    server, id_of_replay=None, strat_id=-1, *, uuid_of_replay=None
):
    if uuid_of_replay is not None:
        warnings.warn(
            "replay arg 'uuid_to_replay` is not actually a uuid, just the unique ID in the DB of the specific run of an experiment. This argument will change soon.",
            DeprecationWarning,
            stacklevel=2,
        )
        id_of_replay = uuid_of_replay

    if id_of_replay is None:
        records = server.db.get_master_records()
        if len(records) > 0:
            id_of_replay = records[-1].unique_id
        else:
            raise RuntimeError("Server has no experiment records!")

    strats = server.db.get_strat_for(id_of_replay, strat_id)
    if strats is not None:
        return strats
    else:
        warnings.warn(
            "No final strat found (likely due to old DB,"
            + " trying to replay tells to generate a final strat. Note"
            + " that this fallback will eventually be removed!",
            DeprecationWarning,
            stacklevel=2,
        )
        # sometimes there's no final strat, e.g. if it's a very old database
        # (we dump strats on crash) in this case, replay the setup and tells
        replay(server, id_of_replay, skip_computations=True)
        # then if the final strat is model-based, refit
        strat = server._strats[strat_id]
        if strat.has_model:
            strat.model.fit(strat.x, strat.y)
        return strat


def get_dataframe_from_replay(
    server, id_of_replay=None, force_replay=False, *, uuid_of_replay=None
):
    if uuid_of_replay is not None:
        warnings.warn(
            "replay arg 'uuid_to_replay` is not actually a uuid, just the unique ID in the DB of the specific run of an experiment. This argument will change soon.",
            DeprecationWarning,
            stacklevel=2,
        )
        id_of_replay = uuid_of_replay

    if id_of_replay is None:
        records = server.db.get_master_records()
        if len(records) > 0:
            id_of_replay = records[-1].unique_id
        else:
            raise RuntimeError("Server has no experiment records!")

    recs = server.db.get_replay_for(id_of_replay)

    strats = get_strats_from_replay(server, id_of_replay, force_replay=force_replay)

    out = pd.DataFrame(
        [flatten_tell_record(server, rec) for rec in recs if rec.message_type == "tell"]
    )

    # flatten any final nested lists
    def _flatten(x):
        return x[0] if len(x) == 1 else x

    for col in out.columns:
        if out[col].dtype == object:
            out.loc[:, col] = out[col].apply(_flatten)

    n_tell_records = len(out)
    n_strat_datapoints = 0
    post_means = []
    post_vars = []

    # collect posterior means and vars
    for strat in strats:
        if strat.has_model:
            post_mean, post_var = strat.predict(strat.x)
            n_tell_records = len(out)
            n_strat_datapoints += len(post_mean)
            post_means.extend(post_mean.detach().numpy())
            post_vars.extend(post_var.detach().numpy())

    if n_tell_records == n_strat_datapoints:
        out["post_mean"] = post_means
        out["post_var"] = post_vars
    else:
        logger.warn(
            f"Number of tell records ({n_tell_records}) does not match "
            + f"number of datapoints in strat ({n_strat_datapoints}) "
            + "cowardly refusing to populate GP mean and var to dataframe!"
        )
    return out
