#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import io
import json
import warnings
from pathlib import Path
from typing import Sequence

import dill
import numpy as np
import torch
from aepsych.config import Config
from aepsych.database.db import Database
from aepsych.database.tables import DBMasterTable, DbReplayTable
from aepsych.utils_logging import getLogger

logger = getLogger()


def combine_dbs(
    out_path: Path | str,
    dbs: list[Path] | Path | str,
    exclude: list[str | Path] | None,
    verbose: bool = True,
) -> int:
    """Combine multiple databases into a single one. If dbs is a list, then
    those dbs will be combined. If it is a single path, we will search for all
    dbs in that directory recursively and combine them.

    Args:
        out_path (Path): Path to the output database.
        dbs (list[Path] | Path): Paths to the input databases.
        exclude (list[str | Path], optional): List of paths to exclude from the
            search. Defaults to None.
        verbose (bool): Whether to log progress. Defaults to True.

    Returns:
        int: Number of experiments combined across all dbs.
    """
    if exclude is None:
        exclude = []

    exclude = [Path(ex) for ex in exclude]

    if isinstance(out_path, str):
        out_path = Path(out_path)
    # If outpath already exists, raise
    if out_path.exists():
        raise FileExistsError(f"Output path {out_path} already exists.")

    # If dbs is a single path, search for all dbs in that directory recursively
    if isinstance(dbs, (Path, str)):
        if isinstance(dbs, str):
            dbs = Path(dbs)
        dbs = list(dbs.glob("**/*.db"))
        dbs.sort()

    dbs = [db for db in dbs if db not in exclude]

    if verbose:
        print(f"Found {len(dbs)} dbs to combine.")

    # Create a new database
    out_db = Database(db_path=str(out_path), update=False)

    # Loop through dbs and add them to out_db
    num_experiments = 0
    for db_path in dbs:
        if verbose:
            print(f"Adding {db_path} to {out_path}")

        db = Database(db_path=str(db_path), update=False, read_only=True)

        # Replay each master
        for master in db.get_master_records():
            transfer_dbs(out_db, master, extra_metadata={"origin_db": str(db_path)})

        db.cleanup()
        num_experiments += 1

    return num_experiments


def transfer_dbs(
    new_db: Database,
    master_record: DBMasterTable,
    extra_metadata: dict[str, str] | None = None,
) -> None:
    """Transfer master record from a db into another db by replaying the setup,
    tell, and ask messages. This will also try to transfer the pickled strategy
    tied to the master record to the new database.

    Args:
        new_db (Database): The new database to transfer to.
        master_record (DBMasterTable): The master record to transfer.
        extra_metadata (dict[str, str], optional): Extra metadata to add to the
            new master record. Defaults to None.

    Returns:
        DBMasterTable: The new master record in the new database.
    """
    print(f"Copying master record with unique ID {master_record.unique_id}")
    new_master = None
    for row in master_record.children_replay:
        # Loop through each row of a master, skips useless rows
        out = _REPLAY_MAP.get(row.message_type, lambda row, new_db, new_master: None)(
            row, new_db, new_master
        )

        if out is not None:
            new_master = out

    if new_master is None:
        warnings.warn(
            "Master record not found even after replay, no strategy will be saved.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Get strategies and save it
    for strat in master_record.children_strat:
        try:
            if isinstance(strat.strat, io.BytesIO):
                strat.strat.seek(0)
                new_db.record_strat(new_master, strat.strat)
            else:
                buffer = io.BytesIO()
                torch.save(strat.strat, buffer, pickle_module=dill)
                buffer.seek(0)
                new_db.record_strat(new_master, buffer)
        except ModuleNotFoundError:
            warnings.warn(
                "Could not save strat to db, it may be incompatible with the current version of aepsych.",
                RuntimeWarning,
                stacklevel=2,
            )

    if extra_metadata is not None:
        # Edit master record to remember the origin db path
        try:
            original_metadata = json.loads(master_record.extra_metadata)
        except (json.decoder.JSONDecodeError, TypeError):
            original_metadata = {}

        original_metadata = extra_metadata | original_metadata
        new_master.extra_metadata = json.dumps(original_metadata)
        new_db._session.add(new_master)
        new_db._session.commit()

    return new_master


def _setup_to_record(
    replay: DbReplayTable, db: Database, master_record: DBMasterTable | None
) -> DBMasterTable:
    # Take a setup row and add it to the db
    config = Config(**replay.message_contents["message"])
    master_record = db.record_setup(
        description=config.get("metadata", "experiment_description", fallback=None),
        name=config.get("metadata", "experiment_name", fallback=None),
        extra_metadata=config.jsonifyMetadata(only_extra=True),
        exp_id=config.get("metadata", "experiment_id", fallback=None),
        request=replay.message_contents,
        par_id=config.get("metadata", "participant_id", fallback=None),
    )
    db.record_config(master_table=master_record, config=config)
    return master_record


def _ask_to_record(
    replay: DbReplayTable, db: Database, master_record: DBMasterTable | None
) -> None:
    # Take an ask row and add it to the db
    db.record_message(
        master_table=master_record,
        type="ask",
        request=replay.message_contents,
    )


def _tell_to_record(
    replay: DbReplayTable, db: Database, master_record: DBMasterTable | None
) -> None:
    # Take a tell row and add it to the db
    db.record_message(
        master_table=master_record,
        type="tell",
        request=replay.message_contents,
    )

    config = replay.message_contents["message"]["config"]
    outcome = replay.message_contents["message"]["outcome"]
    model_data = replay.message_contents["message"].get("model_data", None)
    extra_data = {
        key: value
        for key, value in replay.message_contents["message"].items()
        if key not in ["config", "outcome", "model_data"]
    }
    config_dict = {
        key: value if isinstance(value, (Sequence, np.ndarray)) else [value]
        for key, value in config.items()
    }
    if (
        master_record.children_config[-1].config.getint("common", "stimuli_per_trial")
        > 1
    ):
        # Correct for multi stimuli in old databases
        for key, value in config_dict.items():
            if isinstance(value, list):
                value = np.array(value)
            if value.ndim == 1:
                value = np.expand_dims(value, axis=0)
                config_dict[key] = value

    n_trials = len(list(config_dict.values())[0])

    # Fix outcome to be a dictionary of array-likes
    outcome_tmp = {"outcome": outcome} if not isinstance(outcome, dict) else outcome
    outcome_dict = {
        key: value if isinstance(value, (Sequence, np.ndarray)) else [value]
        for key, value in outcome_tmp.items()
    }

    for i in range(n_trials):  # Go through the trials
        raw_record = db.record_raw(
            master_table=master_record,
            model_data=bool(model_data),
            **extra_data,
        )

        for param_name, param_values in config_dict.items():
            param_value = param_values[i]
            if isinstance(param_value, (Sequence, np.ndarray)):  # Multi stimuli
                for j, v in enumerate(param_value):
                    db.record_param(
                        raw_table=raw_record,
                        param_name=str(param_name) + "_stimuli" + str(j),
                        param_value=str(v),
                    )
            else:  # Single stimuli
                db.record_param(
                    raw_table=raw_record,
                    param_name=str(param_name),
                    param_value=str(param_value),
                )

        # Record outcome
        for outcome_name, outcome_values in outcome_dict.items():
            outcome_value = outcome_values[i]
            db.record_outcome(
                raw_table=raw_record,
                outcome_name=outcome_name,
                outcome_value=float(outcome_value),
            )


_REPLAY_MAP = {
    "setup": _setup_to_record,
    "ask": _ask_to_record,
    "tell": _tell_to_record,
}
