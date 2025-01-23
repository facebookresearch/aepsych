#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
from collections.abc import Iterable
from typing import Dict, Optional, Sequence, Union

import aepsych.utils_logging as utils_logging
import dill
import pandas as pd
import torch

logger = utils_logging.getLogger(logging.INFO)
DEFAULT_DESC = "default description"
DEFAULT_NAME = "default name"


def handle_tell(server, request):
    logger.debug("got tell message!")

    if not server.is_performing_replay:
        server.db.record_message(
            master_table=server._db_master_record, type="tell", request=request
        )
        if "extra_info" in request:
            logger.warning(
                "extra_info should not be used to record extra trial-level data alongside tells, extra data should be added as extra keys in the message."
            )

    # Batch update mode
    if type(request["message"]) == list:
        for msg in request["message"]:
            tell(server, **msg)
    else:
        tell(server, **request["message"])

    if server.strat is not None and server.strat.finished is True:
        logger.info("Recording strat because the experiment is complete.")

        buffer = io.BytesIO()
        torch.save(server.strat, buffer, pickle_module=dill)
        buffer.seek(0)
        server.db.record_strat(master_table=server._db_master_record, strat=buffer)

    return "acq"


def flatten_tell_record(server, rec):
    out = {}
    out["response"] = int(rec.message_contents["message"]["outcome"])

    out.update(
        pd.json_normalize(rec.message_contents["message"]["config"], sep="_").to_dict(
            orient="records"
        )[0]
    )

    if rec.extra_info is not None:
        out.update(rec.extra_info)

    return out


def tell(
    server,
    outcome: Union[Dict[str, Union[str, float]], Sequence],
    config: Dict[str, Union[str, float]],
    model_data: bool = True,
    **extra_data,
):
    """Tell the model which input was run and what the outcome was.

    Arguments:
        server (AEPsychServer): The AEPsych server object.
        outcome (Union[Dict[str, Union[str, float]], Iterable]): The outcome of the trial.
        config (Dict[str, Union[str, float]], optional): A dictionary mapping parameter
            names to values.
        model_data (bool): If True, the data from this trial will be added to the model.
            If False, the trial will be recorded in the db, but will not be modeled.
            Defaults to True.
        **extra_data: Extra info to save to a trial, not modeled.
    """

    if config is None:
        config = {}

    if not server.is_performing_replay:
        _record_tell(server, outcome, config, model_data, **extra_data)

    if model_data:
        x = server._config_to_tensor(config)
        server.strat.add_data(x, outcome)


def _record_tell(
    server,
    outcome: Union[Dict[str, Union[str, float]], Sequence, str],
    config: Dict[str, Union[str, float]],
    model_data: bool = True,
    **extra_data,
):
    server._db_raw_record = server.db.record_raw(
        master_table=server._db_master_record,
        model_data=bool(model_data),
        **extra_data,
    )

    for param_name, param_value in config.items():
        if isinstance(param_value, Iterable) and type(param_value) != str:
            if len(param_value) == 1:
                server.db.record_param(
                    raw_table=server._db_raw_record,
                    param_name=str(param_name),
                    param_value=str(param_value[0]),
                )
            else:
                for i, v in enumerate(param_value):
                    server.db.record_param(
                        raw_table=server._db_raw_record,
                        param_name=str(param_name) + "_stimuli" + str(i),
                        param_value=str(v),
                    )
        else:
            server.db.record_param(
                raw_table=server._db_raw_record,
                param_name=str(param_name),
                param_value=str(param_value),
            )

    if isinstance(outcome, dict):
        for key in outcome.keys():
            server.db.record_outcome(
                raw_table=server._db_raw_record,
                outcome_name=key,
                outcome_value=float(outcome[key]),
            )

    # Check if we get single or multiple outcomes
    # Multiple outcomes come in the form of iterables that aren't strings or single-element tensors
    elif hasattr(outcome, "__iter__") and type(outcome) != str:
        for i, outcome_value in enumerate(outcome):
            if isinstance(outcome_value, Sequence) and type(outcome_value) != str:
                if isinstance(outcome_value, torch.Tensor) and outcome_value.dim() < 2:
                    outcome_value = outcome_value.item()

                elif len(outcome_value) == 1:
                    outcome_value = outcome_value[0]
                else:
                    raise ValueError(
                        "Multi-outcome values must be a list of lists of length 1!"
                    )
            server.db.record_outcome(
                raw_table=server._db_raw_record,
                outcome_name="outcome_" + str(i),
                outcome_value=float(outcome_value),
            )
    else:
        server.db.record_outcome(
            raw_table=server._db_raw_record,
            outcome_name="outcome",
            outcome_value=float(outcome),  # type: ignore
        )
