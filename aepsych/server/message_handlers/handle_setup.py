#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, TypedDict

import aepsych.utils_logging as utils_logging
from aepsych.config import Config
from aepsych.database.data_fetcher import DataFetcher
from aepsych.extensions import ExtensionManager
from aepsych.strategy import SequentialStrategy

logger = utils_logging.getLogger(logging.INFO)

SetupResponse = TypedDict("SetupResponse", {"strat_id": int})


def _configure(server, config: Config) -> SetupResponse:
    """Setup an experiment. This will load extensions from the config then build
    strategies from the config.

    Args:
        server (AEPsychServer): AEPsych server responding to the message.
        config (Config): Config to configure the server with.

    Returns:
        SetupResponse: A dictionary with one entry
            - "strat_id": integer, the stategy ID for what was just set up.
    """
    # Run extension scripts
    server.extensions = ExtensionManager.from_config(config)
    server.extensions.load()

    server._pregen_asks = []  # TODO: Allow each strategy to have its own stack of pre-generated asks

    parnames = config.getlist("common", "parnames", element_type=str)
    server.parnames = parnames
    outcome_types = config.getlist("common", "outcome_types", element_type=str)
    outcome_names = config.getlist(
        "common", "outcome_names", element_type=str, fallback=None
    )
    if outcome_names is None:
        outcome_names = [f"outcome_{i + 1}" for i in range(len(outcome_types))]
    server.outcome_names = outcome_names
    server.config = config
    server.enable_pregen = config.getboolean("common", "pregen_asks", fallback=False)

    server.strat = SequentialStrategy.from_config(config)
    server.strat_id = server.n_strats - 1  # 0-index strats

    for strat in server.strat.strat_list:
        fetcher = DataFetcher.from_config(config, strat.name)
        fetcher.warm_start_strat(server, strat)

    try:
        # Check we have a record to get
        _ = server._db_master_record
    except (
        IndexError
    ):  # We probably don't have a record for this new ID, so we make a dummy
        server._db_master_record = server.db.record_setup()

    return {"strat_id": server.strat_id}


def configure(server, config: Config | None = None, **config_args) -> SetupResponse:
    """Setup an experiment. This function is primarily to preserve backwards
    compatibility. config_args is still usable for unittests and old functions, but if
    config is specified, the server will use that rather than create a new config
    object.

    Args:
        server (AEPsychServer): AEPsych server responding to the message.
        config (Config, optional): Config to configure the server with.
        **config_args: Backwards compat options that wil be turned into a Config object,
            if the config arg is not None, these are ignored entirely.

    Returns:
        SetupResponse: A dictionary with one entry
            - "strat_id": integer, the stategy ID for what was just set up.
    """
    if config is None:
        usedconfig = Config(**config_args)
    else:
        usedconfig = config

    response = _configure(server, usedconfig)
    server.db.record_config(master_table=server._db_master_record, config=usedconfig)
    return response


def handle_setup(server, request: dict[str, Any]) -> SetupResponse:
    """Setup an experiment.

    Args:
        server (AEPsychServer): AEPsych server responding to the message.
        request (dict[str, Any]): A dictionary from the request message, must include
            configuration info in some form.

    Returns:
        SetupResponse: A dictionary with one entry
            - "strat_id": integer, the stategy ID for what was just set up.
    """
    logger.debug("got setup message!")
    ### make a temporary config object to derive parameters because server handles config after table
    if (
        "config_str" in request["message"].keys()
        or "config_dict" in request["message"].keys()
    ):
        tempconfig = Config(**request["message"])
        if not server.is_performing_replay:
            if "metadata" in tempconfig.keys():
                # Get metadata
                exp_name = tempconfig["metadata"].get("experiment_name", fallback=None)
                exp_desc = tempconfig["metadata"].get(
                    "experiment_description", fallback=None
                )
                par_id = tempconfig["metadata"].get("participant_id", fallback=None)

                # This may be populated when replaying
                exp_id = tempconfig["metadata"].get("experiment_id", fallback=None)
                if exp_id is None and server._db_master_record is not None:
                    exp_id = server._db_master_record.experiment_id

                extra_metadata = tempconfig.jsonifyMetadata(only_extra=True)
                server._db_master_record = server.db.record_setup(
                    description=exp_desc,
                    name=exp_name,
                    request=request,
                    exp_id=exp_id,
                    par_id=par_id,
                    extra_metadata=extra_metadata if extra_metadata != "" else None,
                )
            else:  # No metadata set, still record the master
                exp_id = (
                    server._db_master_record.experiment_id
                    if server._db_master_record is not None
                    else None
                )

                server._db_master_record = server.db.record_setup(
                    request=request, exp_id=exp_id
                )

        strat_id = configure(server, config=tempconfig)
    else:
        raise RuntimeError("Missing a configure message!")

    return strat_id
