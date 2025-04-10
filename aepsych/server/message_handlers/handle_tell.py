#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
from typing import Any, Sequence, TypeAlias, TypedDict

import aepsych.utils_logging as utils_logging
import dill
import numpy as np
import pandas as pd
import torch
from aepsych.database.tables import DbReplayTable

logger = utils_logging.getLogger(logging.INFO)
DEFAULT_DESC = "default description"
DEFAULT_NAME = "default name"

TellResponse = TypedDict(
    "TellResponse", {"trials_recorded": int, "model_data_added": int}
)

# This form is deprecated in 3.12 in favor of a "type statement", but we support 3.10+
OutcomeType: TypeAlias = (
    dict[
        str,
        str | float | Sequence[str | float] | np.ndarray,
    ]
    | Sequence
    | float
    | str
)
ParameterConfigType: TypeAlias = dict[
    str,
    str | float | Sequence[str | float | Sequence[str | float]] | np.ndarray,
]

# Annoyingly, arrays are like sequences but they aren't sequences
ARRAY_LIKE = (Sequence, np.ndarray)


def handle_tell(server, request: dict[str, Any]) -> TellResponse:
    """Tell the model which input was run and what the outcome was.

    Args:
        server (AEPsychServer): The AEPsych server object.
        request (dict[str, Any]): A dictionary from the request message, must include
            tell data.

    Returns:
        TellResponse: A dictionary with these entries
            - "trials_recorded": integer, the number of trials recorded in the
                database.
            - "model_data_added": integer, the number of datapoints added to the model.
    """
    logger.debug("got tell message!")

    if not server.is_performing_replay:
        server.db.record_message(
            master_table=server._db_master_record, type="tell", request=request
        )
        if "extra_info" in request:
            logger.warning(
                "extra_info should not be used to record extra trial-level data alongside tells, extra data should be added as extra keys in the message."
            )

    tell_response = tell(server, **request["message"])

    if server.strat is not None and server.strat.finished is True:
        logger.info("Recording strat because the experiment is complete.")

        buffer = io.BytesIO()
        torch.save(server.strat, buffer, pickle_module=dill)
        buffer.seek(0)
        server.db.record_strat(master_table=server._db_master_record, strat=buffer)

    return tell_response


def flatten_tell_record(server, rec: DbReplayTable) -> dict[str, Any]:
    """Flatten a tell replay record into a dictionary including the trial parameter
    configuration and the response.

    Args:
        server (AEPsychServer): The AEPsych server object. This is currently ignored.
        rec (DbReplayTable): The replay table record. This must be a tell record.

    Returns:
        dict[str, Any]: A dictionary including the information from the tell replay
            record.
    """
    out = {}
    out["response"] = int(rec.message_contents["message"]["outcome"])

    out.update(
        pd.json_normalize(rec.message_contents["message"]["config"], sep="_").to_dict(
            orient="records"
        )[0]
    )

    extra_data = {
        key: value
        for key, value in rec.message_contents["message"].items()
        if key not in ["config", "outcome", "model_data"]
    }
    if len(extra_data) != 0:
        out.update(extra_data)

    return out


def tell(
    server,
    outcome: OutcomeType,
    config: ParameterConfigType | None = None,
    model_data: bool = True,
    **extra_data,
) -> TellResponse:
    """Tell the model which input was run and what the outcome was.

    Args:
        server (AEPsychServer): The AEPsych server object.
        outcome (OutcomeType): The outcome of the trial. If it's a float, it's a single
            trial single outcome. If it's a sequence, it is multiple trials single
            outcome. If it is a dictionary, it is a multi outcome with the same rules
            for float vs sequence of floats for single and multi trials respeectively.
        config (ParameterConfigType, optional): A dictionary mapping
            parameter names to values. Each key is a parameter, its value is either an
            iterable (if it represents multiple trials or multi stimuli) or a non-
            iterable for a single trial single stimuli case.
        model_data (bool): If True, the data from this trial will be added to the model.
            If False, the trial will be recorded in the db, but will not be modeled.
            Defaults to True.
        **extra_data: Extra info to save to a trial, not modeled.

    Returns:
        TellResponse: A dictionary with these entries
            - "trials_recorded": integer, the number of trials recorded in the
                database.
            - "model_data_added": integer, the number of datapoints added to the model.
    """
    tell_response: TellResponse = {"trials_recorded": 0, "model_data_added": 0}
    if config is None:
        config = {}

    if not server.is_performing_replay:
        tell_response["trials_recorded"] = _record_tell(
            server, outcome, config, model_data, **extra_data
        )

    if model_data:
        x = server._config_to_tensor(config)

        if isinstance(outcome, dict):
            # Make sure the outcome keys match outcome names exactly
            if not set(outcome.keys()) == set(server.outcome_names):
                raise KeyError(
                    f"Outcome keys {outcome.keys()} do not match outcome names {server.outcome_names}."
                )

            values = []
            for outcome_name in server.outcome_names:
                value = outcome[outcome_name]
                if isinstance(value, (str, float, int)):
                    value = [float(value)]

                value_tensor = torch.tensor(value)
                if value_tensor.ndim == 1:
                    value_tensor = value_tensor.unsqueeze(-1)

                values.append(value_tensor)

            server.strat.add_data(x, torch.hstack(values))
        else:
            server.strat.add_data(x, outcome)

        tell_response["model_data_added"] = len(x)

    return tell_response


def _record_tell(
    server,
    outcome: OutcomeType,
    config: ParameterConfigType,
    model_data: bool = True,
    **extra_data,
) -> int:
    """Records the data from a tell into the server. Each trial is written as a separate
    row in the raw table.

    Args:
        server (AEPsychServer): AEPsych server that has the database to write to.
        outcome (OutcomeType): The outcomes to write to the database.
        config (ParamterConfigType): The parameter config dictionary to write to the
            database.
        model_data (bool): Whether or not this data was given to the model.
        **extra_data: Extra data to write to the raw table.

    Returns:
        int: The number of trials (rows in the raw table) written to the database.
    """
    config_dict = {
        key: value if isinstance(value, ARRAY_LIKE) else [value]
        for key, value in config.items()
    }
    n_trials = len(list(config_dict.values())[0])

    # Fix outcome to be a dictionary of array-likes
    outcome_tmp = {"outcome": outcome} if not isinstance(outcome, dict) else outcome
    outcome_dict = {
        key: value if isinstance(value, ARRAY_LIKE) else [value]
        for key, value in outcome_tmp.items()
    }

    for i in range(n_trials):  # Go through the trials
        server._db_raw_record = server.db.record_raw(
            master_table=server._db_master_record,
            model_data=bool(model_data),
            **extra_data,
        )

        for param_name, param_values in config_dict.items():
            param_value = param_values[i]
            if isinstance(param_value, ARRAY_LIKE):  # Multi stimuli
                for j, v in enumerate(param_value):
                    server.db.record_param(
                        raw_table=server._db_raw_record,
                        param_name=str(param_name) + "_stimuli" + str(j),
                        param_value=str(v),
                    )
            else:  # Single stimuli
                server.db.record_param(
                    raw_table=server._db_raw_record,
                    param_name=str(param_name),
                    param_value=str(param_value),
                )

        # Record outcome
        for outcome_name, outcome_values in outcome_dict.items():
            outcome_value = outcome_values[i]
            server.db.record_outcome(
                raw_table=server._db_raw_record,
                outcome_name=outcome_name,
                outcome_value=float(outcome_value),
            )

    return n_trials
