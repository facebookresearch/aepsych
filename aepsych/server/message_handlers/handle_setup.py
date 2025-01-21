#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import aepsych.utils_logging as utils_logging
from aepsych.config import Config
from aepsych.database.data_fetcher import DataFetcher
from aepsych.extensions import ExtensionManager
from aepsych.strategy import SequentialStrategy
from aepsych.version import __version__

logger = utils_logging.getLogger(logging.INFO)


def _configure(server, config):
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

    return server.strat_id


def configure(server, config=None, **config_args):
    # To preserve backwards compatibility, config_args is still usable for unittests and old functions.
    # But if config is specified, the server will use that rather than create a new config object.
    if config is None:
        usedconfig = Config(**config_args)
    else:
        usedconfig = config
    if "experiment" in usedconfig:
        logger.warning(
            'The "experiment" section is being deprecated from configs. Please put everything in the "experiment" section in the "common" section instead.'
        )

        for i in usedconfig["experiment"]:
            usedconfig["common"][i] = usedconfig["experiment"][i]
        del usedconfig["experiment"]

    version = usedconfig.version
    if version < __version__:
        try:
            usedconfig.convert_to_latest()

            server.db.perform_updates()
            logger.warning(
                f"Config version {version} is less than AEPsych version {__version__}. The config was automatically modified to be compatible. Check the config table in the db to see the changes."
            )
        except RuntimeError:
            logger.warning(
                f"Config version {version} is less than AEPsych version {__version__}, but couldn't automatically update the config! Trying to configure the server anyway..."
            )

    strat_id = _configure(server, usedconfig)
    server.db.record_config(master_table=server._db_master_record, config=usedconfig)
    return strat_id


def handle_setup(server, request):
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
