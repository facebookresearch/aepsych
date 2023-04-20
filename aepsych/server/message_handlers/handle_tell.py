# LICENSE file in the root directory of this source tree.
import io
import logging
from collections.abc import Iterable

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


def tell(server, outcome, config, model_data=True):
    """tell the model which input was run and what the outcome was
    Arguments:
        inputs {dict} -- dictionary, keys are strings, values are floats or int.
        keys should inclde all of the parameters we are tuning over, plus 'outcome'
        which would be in {0, 1}.
        TODO better types
    """
    if not server.is_performing_replay:
        server._db_raw_record = server.db.record_raw(
            master_table=server._db_master_record,
            model_data=bool(model_data),
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

        # Check if we get single or multiple outcomes
        # Multiple outcomes come in the form of iterables that aren't strings or single-element tensors
        if isinstance(outcome, Iterable) and type(outcome) != str:
            for i, outcome_value in enumerate(outcome):
                if isinstance(outcome_value, Iterable) and type(outcome_value) != str:
                    if (
                        isinstance(outcome_value, torch.Tensor)
                        and outcome_value.dim() < 2
                    ):
                        outcome_value = outcome_value.item()

                    elif len(outcome_value) == 1:
                        outcome_value = outcome_value[0]
                    else:
                        raise ValueError(
                            "Multi-outcome values must be a list of lists of length 1!"
                        )
                print(outcome_value)
                server.db.record_outcome(
                    raw_table=server._db_raw_record,
                    outcome_name="outcome_" + str(i),
                    outcome_value=float(outcome_value),
                )
        else:
            server.db.record_outcome(
                raw_table=server._db_raw_record,
                outcome_name="outcome",
                outcome_value=float(outcome),
            )

    if model_data:
        if not server.use_ax:
            x = server._config_to_tensor(config)
        else:
            x = config
        server.strat.add_data(x, outcome)
