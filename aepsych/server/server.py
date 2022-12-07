#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import io
import logging
import os
import sys
import threading
import traceback
import warnings
from collections.abc import Iterable

import aepsych.database.db as db
import aepsych.utils_logging as utils_logging
import dill
import numpy as np
import pandas as pd
import torch
from aepsych.config import Config
from aepsych.server.sockets import BAD_REQUEST, DummySocket, createSocket
from aepsych.strategy import SequentialStrategy
from aepsych.version import __version__

logger = utils_logging.getLogger(logging.INFO)
DEFAULT_DESC = "default description"
DEFAULT_NAME = "default name"


def get_next_filename(folder, fname, ext):
    n = sum(1 for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
    return f"{folder}/{fname}_{n+1}.{ext}"


class AEPsychServer(object):
    def __init__(self, socket=None, database_path=None, thrift=False):
        """Server for doing black box optimization using gaussian processes.
        Keyword Arguments:
            socket -- socket object that implements `send` and `receive` for json
            messages (default: DummySocket()).
            TODO actually make an abstract interface to subclass from here
        """
        if socket is None:
            self.socket = DummySocket()
        else:
            self.socket = socket
        self.db = None
        self.is_performing_replay = False
        self.exit_server_loop = False
        self._db_master_record = None
        self._db_raw_record = None
        self.db = db.Database(database_path)
        self.skip_computations = False

        if self.db.is_update_required():
            raise RuntimeError(
                f'The database needs to be updated. You can perform the update by running "python3 aepsych/server/server.py database --update --d {database_path}"'
            )

        self._strats = []
        self._parnames = []
        self._configs = []
        self.strat_id = -1

        self.debug = False
        self.is_using_thrift = thrift
        self.receive_thread = threading.Thread(
            target=self._receive_send, args=(self.exit_server_loop,), daemon=True
        )

        self.queue = []

    def cleanup(self):
        self.socket.close()

    def _receive_send(self, is_exiting):
        while True:
            request = self.socket.receive(is_exiting)
            if request != BAD_REQUEST:
                self.queue.append(request)
            if self.exit_server_loop:
                break
        logger.info("Terminated input thread")

    def _handle_queue(self):
        if self.queue:
            request = self.queue.pop(0)
            try:
                result = self.handle_request(request)
            except Exception as e:
                error_message = f"Request '{request}' raised error '{e}'!"
                result = f"server_error, {error_message}"
                logger.error(f"{error_message}! Full traceback follows:")
                logger.error(traceback.format_exc())
            self.socket.send(result)

    def serve(self):
        """Run the server. Note that all configuration outside of socket type and port
        happens via messages from the client. The server simply forwards messages from
        the client to its `setup`, `ask` and `tell` methods, and responds with either
        acknowledgment or other response as needed. To understand the server API, see
        the docs on the methods in this class.
        Raises:
            RuntimeError: if a request from a client has no request type
            RuntimeError: if a request from a client has no known request type
            TODO make things a little more robust to bad messages from client; this
             requires resetting the req/rep queue status.
        """
        logger.info("Server up, waiting for connections!")
        logger.info("Ctrl-C to quit!")
        # yeah we're not sanitizing input at all

        # Start the method to accept a client connection

        if self.is_using_thrift is True:
            self.queue.append(self.socket.receive())
            self._handle_queue()
        else:
            self.socket.accept_client()
            self.receive_thread.start()
            while True:
                self._handle_queue()
                if self.exit_server_loop:
                    break
            # Close the socket and terminate with code 0
            self.cleanup()
            sys.exit(0)

    def replay(self, uuid_to_replay, skip_computations=False):
        """
        Run a replay against the server. The UUID will be looked up in the database.
        if skip_computations is true, skip all the asks and queries, which should make the replay much faster.
        """
        if uuid_to_replay is None:
            raise RuntimeError("UUID is a required parameter to perform a replay")

        if self.db is None:
            raise RuntimeError("A database is required to perform a replay")

        if skip_computations is True:
            logger.info(
                "skip_computations=True, make sure to refit the final strat before doing anything!"
            )

        master_record = self.db.get_master_record(uuid_to_replay)

        if master_record is None:
            raise RuntimeError(
                f"The UUID {uuid_to_replay} isn't in the database. Unable to perform replay."
            )

        # this prevents writing back to the DB and creating a circular firing squad
        self.is_performing_replay = True
        self.skip_computations = skip_computations

        for result in master_record.children_replay:
            request = result.message_contents
            logger.debug(f"replay - type = {result.message_type} request = {request}")
            self.handle_request(request)

        self._db_master_record = master_record
        self.is_performing_replay = False
        self.skip_computations = False

    def _unpack_strat_buffer(self, strat_buffer):
        if isinstance(strat_buffer, io.BytesIO):
            strat = torch.load(strat_buffer, pickle_module=dill)
            strat_buffer.seek(0)
        elif isinstance(strat_buffer, bytes):
            warnings.warn(
                "Strat buffer is not in bytes format!"
                + " This is a deprecated format, loading using dill.loads.",
                DeprecationWarning,
            )
            strat = dill.loads(strat_buffer)
        else:
            raise RuntimeError("Trying to load strat in unknown format!")
        return strat

    def get_strats_from_replay(self, uuid_of_replay=None, force_replay=False):
        if uuid_of_replay is None:
            records = self.db.get_master_records()
            if len(records) > 0:
                uuid_of_replay = records[-1].experiment_id
            else:
                raise RuntimeError("Server has no experiment records!")

        if force_replay:
            warnings.warn(
                "Force-replaying to get non-final strats is deprecated after the ability"
                + " to save all strats was added, and will eventually be removed.",
                DeprecationWarning,
            )
            self.replay(uuid_of_replay, skip_computations=True)
            for strat in self._strats:
                if strat.has_model:
                    strat.model.fit(strat.x, strat.y)
            return self._strats
        else:
            strat_buffers = self.db.get_strats_for(uuid_of_replay)
            return [self._unpack_strat_buffer(sb) for sb in strat_buffers]

    def get_strat_from_replay(self, uuid_of_replay=None, strat_id=-1):
        if uuid_of_replay is None:
            records = self.db.get_master_records()
            if len(records) > 0:
                uuid_of_replay = records[-1].experiment_id
            else:
                raise RuntimeError("Server has no experiment records!")

        strat_buffer = self.db.get_strat_for(uuid_of_replay, strat_id)
        if strat_buffer is not None:
            return self._unpack_strat_buffer(strat_buffer)
        else:
            warnings.warn(
                "No final strat found (likely due to old DB,"
                + " trying to replay tells to generate a final strat. Note"
                + " that this fallback will eventually be removed!",
                DeprecationWarning,
            )
            # sometimes there's no final strat, e.g. if it's a very old database
            # (we dump strats on crash) in this case, replay the setup and tells
            self.replay(uuid_of_replay, skip_computations=True)
            # then if the final strat is model-based, refit
            strat = self._strats[strat_id]
            if strat.has_model:
                strat.model.fit(strat.x, strat.y)
            return strat

    def _flatten_tell_record(self, rec):
        out = {}
        out["response"] = int(rec.message_contents["message"]["outcome"])

        out.update(
            pd.json_normalize(
                rec.message_contents["message"]["config"], sep="_"
            ).to_dict(orient="records")[0]
        )

        if rec.extra_info is not None:
            out.update(rec.extra_info)

        return out

    def get_dataframe_from_replay(self, uuid_of_replay=None, force_replay=False):
        # DeprecationWarning
        warnings.warn(
            "get_dataframe_from_replay is deprecated."
            + " Use generate_experiment_table with return_df = True instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if uuid_of_replay is None:
            records = self.db.get_master_records()
            if len(records) > 0:
                uuid_of_replay = records[-1].experiment_id
            else:
                raise RuntimeError("Server has no experiment records!")

        recs = self.db.get_replay_for(uuid_of_replay)

        strats = self.get_strats_from_replay(uuid_of_replay, force_replay=force_replay)

        out = pd.DataFrame(
            [
                self._flatten_tell_record(rec)
                for rec in recs
                if rec.message_type == "tell"
            ]
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

    def generate_experiment_table(
        self, experiment_id, table_name="experiment_table", return_df=False
    ):
        """Generate a table of a given experiment with all the raw data.

        This table is generated from the database, and is added to the
        experiment's database.

        Args:
            experiment_id (str): The experiment ID to generate the table for.
            table_name (str): The name of the table. Defaults to
                "experiment_table".
            return_df (bool): If True, also return the dataframe.

        Returns:
            pd.DataFrame: The dataframe of the experiment table, if
                return_df is True.
        """
        param_space = self.db.get_param_for(experiment_id, 1)
        outcome_space = self.db.get_outcome_for(experiment_id, 1)

        columns = []
        columns.append("iteration_id")
        for param in param_space:
            columns.append(param.param_name)
        for outcome in outcome_space:
            columns.append(outcome.outcome_name)

        columns.append("timestamp")

        # Create dataframe
        df = pd.DataFrame(columns=columns)

        # Fill dataframe
        for raw in self.db.get_raw_for(experiment_id):
            row = {}
            row["iteration_id"] = raw.unique_id
            for param in raw.children_param:
                row[param.param_name] = param.param_value
            for outcome in raw.children_outcome:
                row[outcome.outcome_name] = outcome.outcome_value
            row["timestamp"] = raw.timestamp
            # concat to dataframe
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # Make iteration_id the index
        df.set_index("iteration_id", inplace=True)

        # Save to .db file
        df.to_sql(table_name, self.db.get_engine(), if_exists="replace")

        if return_df:
            return df

    def versioned_handler(self, request):
        handled_types = ["setup", "resume", "ask"]
        if request["type"] == "setup":
            if request["version"] == "0.01":
                ret_val = self.handle_setup_v01(request)
            else:
                raise RuntimeError(
                    f"Unknown message version {request['version']} for message 'setup'!"
                )
        elif request["type"] == "resume":
            if request["version"] == "0.01":
                ret_val = self.handle_resume_v01(request)
            else:
                raise RuntimeError(
                    f"Unknown message version {request['version']} for message 'resume'!"
                )
        elif request["type"] == "ask":
            if request["version"] == "0.01":
                ret_val = self.handle_ask_v01(request)
            else:
                raise RuntimeError(
                    f"Unknown message version {request['version']} for message 'ask'!"
                )
        if request["type"] in handled_types:
            logger.info(f"Received msg [{request['type']}]")

        else:
            warnings.warn(
                "Got versioned handler but no version, falling back to unversioned!"
            )
            return self.unversioned_handler(request)
        return ret_val

    def handle_setup_v01(self, request):
        logger.debug("got setup message!")
        ### make a temporary config object to derive parameters because server handles config after table
        if (
            "config_str" in request["message"].keys()
            or "config_dict" in request["message"].keys()
        ):
            tempconfig = Config(**request["message"])
            if not self.is_performing_replay:
                experiment_id = None
                if self._db_master_record is not None:
                    experiment_id = self._db_master_record.experiment_id
                if "metadata" in tempconfig.keys():
                    cdesc = (
                        tempconfig["metadata"]["experiment_description"]
                        if ("experiment_description" in tempconfig["metadata"].keys())
                        else DEFAULT_DESC
                    )
                    cname = (
                        tempconfig["metadata"]["experiment_name"]
                        if ("experiment_name" in tempconfig["metadata"].keys())
                        else DEFAULT_NAME
                    )
                    cid = (
                        tempconfig["metadata"]["experiment_id"]
                        if ("experiment_id" in tempconfig["metadata"].keys())
                        else None
                    )
                    self._db_master_record = self.db.record_setup(
                        description=cdesc,
                        name=cname,
                        request=request,
                        id=cid,
                        extra_metadata=tempconfig.jsonifyMetadata(),
                    )
                ### if the metadata does not exist, we are going to log nothing
                else:
                    self._db_master_record = self.db.record_setup(
                        description=DEFAULT_DESC,
                        name=DEFAULT_NAME,
                        request=request,
                        id=experiment_id,
                    )

            strat_id = self.configure(config=tempconfig)
        else:
            raise RuntimeError("Missing a configure message!")

        return strat_id

    def handle_resume_v01(self, request):
        logger.debug("got resume message!")
        strat_id = int(request["message"]["strat_id"])
        self.strat_id = strat_id
        if not self.is_performing_replay:
            self.db.record_message(
                master_table=self._db_master_record, type="resume", request=request
            )
        return self.strat_id

    def handle_ask_v01(self, request):
        """Returns dictionary with two entries:
        "config" -- dictionary with config (keys are strings, values are floats)
        "is_finished" -- bool, true if the strat is finished
        """
        logger.debug("got ask message!")
        new_config = {"config": self.ask(), "is_finished": self.strat.finished}
        if not self.is_performing_replay:
            self.db.record_message(
                master_table=self._db_master_record, type="ask", request=request
            )
        return new_config

    def unversioned_handler(self, request):
        message_map = {
            "setup": self.handle_setup,
            "ask": self.handle_ask,
            "tell": self.handle_tell,
            "update": self.handle_update,
            "query": self.handle_query,
            "parameters": self.handle_params,
            "can_model": self.handle_can_model,
            "exit": self.handle_exit,
            "get_config": self.handle_get_config,
            "finish_strategy": self.handle_finish_strategy,
            "strategy_name": self.handle_strategy_name,
        }

        if "type" not in request.keys():
            raise RuntimeError(f"Request {request} contains no request type!")
        else:
            type = request["type"]
            if type in message_map.keys():
                logger.info(f"Received msg [{type}]")
                ret_val = message_map[type](request)

                return ret_val
            else:
                exception_message = (
                    f"unknown type: {type}. Allowed types [{message_map.keys()}]"
                )

                raise RuntimeError(exception_message)

    def handle_setup(self, request):
        logger.debug("got setup message!")

        if not self.is_performing_replay:
            experiment_id = None
            if self._db_master_record is not None:
                experiment_id = self._db_master_record.experiment_id

            self._db_master_record = self.db.record_setup(
                description=DEFAULT_DESC,
                name=DEFAULT_NAME,
                request=request,
                id=experiment_id,
            )

        if (
            "config_str" in request["message"].keys()
            or "config_dict" in request["message"].keys()
        ):

            _ = self.configure(**request["message"])
        else:
            raise RuntimeError("Missing a configure message!")
        new_config = self.handle_ask(request)

        return new_config

    def handle_ask(self, request):
        logger.debug("got ask message!")
        new_config = self.ask()

        if not self.is_performing_replay:
            self.db.record_message(
                master_table=self._db_master_record, type="ask", request=request
            )

        return new_config

    def handle_tell(self, request):
        logger.debug("got tell message!")

        if not self.is_performing_replay:
            self.db.record_message(
                master_table=self._db_master_record, type="tell", request=request
            )

        # Batch update mode
        if type(request["message"]) == list:
            for msg in request["message"]:
                self.tell(**msg)
        else:
            self.tell(**request["message"])

        if self.strat is not None and self.strat.finished is True:
            logger.info("Recording strat because the experiment is complete.")

            buffer = io.BytesIO()
            torch.save(self.strat, buffer, pickle_module=dill)
            buffer.seek(0)
            self.db.record_strat(master_table=self._db_master_record, strat=buffer)

        return "acq"

    def handle_update(self, request):
        # update is syntactic sugar for tell, then ask
        logger.debug("got update message!")

        self.handle_tell(request)

        new_config = self.handle_ask(request)

        return new_config

    def handle_params(self, request):
        logger.debug("got parameters message!")
        if not self.is_performing_replay:
            self.db.record_message(
                master_table=self._db_master_record, type="parameters", request=request
            )
        config_setup = {
            self.parnames[i]: [self.strat.lb[i].item(), self.strat.ub[i].item()]
            for i in range(len(self.parnames))
        }
        return config_setup

    def handle_can_model(self, request):
        # Check if the strategy has finished initialization; i.e.,
        # if it has a model and data to fit (strat.can_fit)
        logger.debug("got can_model message!")
        if not self.is_performing_replay:
            self.db.record_message(
                master_table=self._db_master_record, type="can_model", request=request
            )
        return {"can_model": self.strat.can_fit}

    def handle_query(self, request):
        logger.debug("got query message!")
        if not self.is_performing_replay:
            self.db.record_message(
                master_table=self._db_master_record, type="query", request=request
            )
        response = self.query(**request["message"])
        return response

    def query(
        self,
        query_type="max",
        probability_space=False,
        x=None,
        y=None,
        constraints=None,
    ):
        if self.skip_computations:
            return None

        constraints = constraints or {}
        response = {
            "query_type": query_type,
            "probability_space": probability_space,
            "constraints": constraints,
        }
        if query_type == "max":
            fmax, fmax_loc = self.strat.get_max(constraints)
            response["y"] = fmax.item()
            response["x"] = self._tensor_to_config(fmax_loc)
        elif query_type == "min":
            fmin, fmin_loc = self.strat.get_min(constraints)
            response["y"] = fmin.item()
            response["x"] = self._tensor_to_config(fmin_loc)
        elif query_type == "prediction":
            # returns the model value at x
            if x is None:  # TODO: ensure if x is between lb and ub
                raise RuntimeError("Cannot query model at location = None!")
            mean, var = self.strat.predict(
                self._config_to_tensor(x).unsqueeze(axis=0),
                probability_space=probability_space,
            )
            response["x"] = x
            response["y"] = mean.item()
        elif query_type == "inverse":
            # expect constraints to be a dictionary; values are float arrays size 1 (exact) or 2 (upper/lower bnd)
            constraints = {self.parnames.index(k): v for k, v in constraints.items()}
            nearest_y, nearest_loc = self.strat.inv_query(
                y, constraints, probability_space=probability_space
            )
            response["y"] = nearest_y
            response["x"] = self._tensor_to_config(nearest_loc)
        else:
            raise RuntimeError("unknown query type!")
        # ensure all x values are arrays
        response["x"] = {
            k: np.array([v]) if np.array(v).ndim == 0 else v
            for k, v in response["x"].items()
        }
        return response

    def handle_exit(self, request):
        # Make local server write strats into DB and close the connection
        termination_type = "Normal termination"
        logger.info("Got termination message!")
        self.write_strats(termination_type)
        if not self.is_using_thrift:
            self.exit_server_loop = True

        # If using thrift, it will add 'Terminate' to the queue and pass it to thrift server level
        return "Terminate"

    def handle_get_config(self, request):
        msg = request["message"]
        section = msg.get("section", None)
        prop = msg.get("property", None)

        # If section and property are not specified, return the whole config
        if section is None and prop is None:
            return self.config.to_dict(deduplicate=False)

        # If section and property are not both specified, raise an error
        if section is None and prop is not None:
            raise RuntimeError("Message contains a property but not a section!")
        if section is not None and prop is None:
            raise RuntimeError("Message contains a section but not a property!")

        # If both section and property are specified, return only the relevant value from the config
        return self.config.to_dict(deduplicate=False)[section][prop]

    def handle_finish_strategy(self, request):
        self.strat.finish()
        return f"finished strategy {self.strat.name}"

    def handle_strategy_name(self, request):
        return self.strat.name

    ### Properties that are set on a per-strat basis
    @property
    def strat(self):
        if self.strat_id == -1:
            return None
        else:
            return self._strats[self.strat_id]

    @strat.setter
    def strat(self, s):
        self._strats.append(s)

    @property
    def config(self):
        if self.strat_id == -1:
            return None
        else:
            return self._configs[self.strat_id]

    @config.setter
    def config(self, s):
        self._configs.append(s)

    @property
    def parnames(self):
        if self.strat_id == -1:
            return []
        else:
            return self._parnames[self.strat_id]

    @parnames.setter
    def parnames(self, s):
        self._parnames.append(s)

    @property
    def n_strats(self):
        return len(self._strats)

    def ask(self):
        """get the next point to query from the model
        Returns:
            dict -- new config dict (keys are strings, values are floats)
        """
        if self.skip_computations:
            # HACK to makke sure strategies finish correctly
            self.strat._strat._count += 1
            if self.strat._strat.finished:
                self.strat._make_next_strat()
            return None

        # index by [0] is temporary HACK while serverside
        # doesn't handle batched ask
        next_x = self.strat.gen()[0]
        return self._tensor_to_config(next_x)

    def _tensor_to_config(self, next_x):
        config = {}
        for name, val in zip(self.parnames, next_x):
            if val.dim() == 0:
                config[name] = [float(val)]
            else:
                config[name] = np.array(val)
        return config

    def _config_to_tensor(self, config):
        unpacked = [config[name] for name in self.parnames]

        # handle config elements being either scalars or length-1 lists
        if isinstance(unpacked[0], list):
            x = torch.tensor(np.stack(unpacked, axis=0)).squeeze(-1)
        else:
            x = torch.tensor(np.stack(unpacked))
        return x

    def tell(self, outcome, config, model_data=True):
        """tell the model which input was run and what the outcome was
        Arguments:
            inputs {dict} -- dictionary, keys are strings, values are floats or int.
            keys should inclde all of the parameters we are tuning over, plus 'outcome'
            which would be in {0, 1}.
            TODO better types
        """
        if not self.is_performing_replay:
            self._db_raw_record = self.db.record_raw(
                master_table=self._db_master_record,
                model_data=bool(model_data),
            )

            for param_name, param_value in config.items():
                if isinstance(param_value, Iterable) and type(param_value) != str:
                    if len(param_value) == 1:
                        self.db.record_param(
                            raw_table=self._db_raw_record,
                            param_name=str(param_name),
                            param_value=str(param_value[0]),
                        )
                    else:
                        for i, v in enumerate(param_value):
                            self.db.record_param(
                                raw_table=self._db_raw_record,
                                param_name=str(param_name) + "_stimuli" + str(i),
                                param_value=str(v),
                            )
                else:
                    self.db.record_param(
                        raw_table=self._db_raw_record,
                        param_name=str(param_name),
                        param_value=str(param_value),
                    )

            # Check if we get single or multiple outcomes
            # Multiple outcomes come in the form of iterables that aren't strings or single-element tensors
            if isinstance(outcome, Iterable) and type(outcome) != str:
                for i, outcome_value in enumerate(outcome):
                    if (
                        isinstance(outcome_value, Iterable)
                        and type(outcome_value) != str
                    ):
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
                    self.db.record_outcome(
                        raw_table=self._db_raw_record,
                        outcome_name="outcome_" + str(i),
                        outcome_value=float(outcome_value),
                    )
            else:
                self.db.record_outcome(
                    raw_table=self._db_raw_record,
                    outcome_name="outcome",
                    outcome_value=float(outcome),
                )

        if model_data:
            x = self._config_to_tensor(config)
            y = outcome
            self.strat.add_data(x, y)

    def _configure(self, config):
        self.parnames = config._str_to_list(
            config.get("common", "parnames"), element_type=str
        )
        self.config = config
        self.strat = SequentialStrategy.from_config(config)
        self.strat_id = self.n_strats - 1  # 0-index strats
        return self.strat_id

    def configure(self, config=None, **config_args):
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

                self.db.perform_updates()
                logger.warning(
                    f"Config version {version} is less than AEPsych version {__version__}. The config was automatically modified to be compatible. Check the config table in the db to see the changes."
                )
            except RuntimeError:
                logger.warning(
                    f"Config version {version} is less than AEPsych version {__version__}, but couldn't automatically update the config! Trying to configure the server anyway..."
                )

        self.db.record_config(master_table=self._db_master_record, config=usedconfig)
        return self._configure(usedconfig)

    def __getstate__(self):
        # nuke the socket since it's not pickleble
        state = self.__dict__.copy()
        del state["socket"]
        del state["db"]
        return state

    def write_strats(self, termination_type):
        if self._db_master_record is not None and self.strat is not None:
            logger.info(f"Dumping strats to DB due to {termination_type}.")
            for strat in self._strats:
                buffer = io.BytesIO()
                torch.save(strat, buffer, pickle_module=dill)
                buffer.seek(0)
                self.db.record_strat(master_table=self._db_master_record, strat=buffer)

    def generate_debug_info(self, exception_type, dumptype):
        fname = get_next_filename(".", dumptype, "pkl")
        logger.exception(f"Got {exception_type}, exiting! Server dump in {fname}")
        dill.dump(self, open(fname, "wb"))

    def handle_request(self, request):
        if "version" in request.keys():
            return self.versioned_handler(request)

        return self.unversioned_handler(request)


def startServerAndRun(
    server_class, socket=None, database_path=None, config_path=None, uuid_of_replay=None
):
    server = server_class(socket=socket, database_path=database_path)
    try:
        if config_path is not None:
            with open(config_path) as f:
                config_str = f.read()
            server.configure(config_str=config_str)

        if socket is not None:
            if uuid_of_replay is not None:
                server.replay(uuid_of_replay, skip_computations=True)
                server._db_master_record = server.db.get_master_record(uuid_of_replay)
            server.serve()
        else:
            if config_path is not None:
                logger.info(
                    "You have passed in a config path but this is a replay. If there's a config in the database it will be used instead of the passed in config path."
                )
            server.replay(uuid_of_replay)
    except (KeyboardInterrupt):
        exception_type = "CTRL+C"
        dump_type = "dump"
        server.write_strats(exception_type)
        server.generate_debug_info(exception_type, dump_type)
    except RuntimeError as e:
        exception_type = "RuntimeError"
        dump_type = "crashdump"
        server.write_strats(exception_type)
        server.generate_debug_info(exception_type, dump_type)
        raise RuntimeError(e)


def parse_argument():
    parser = argparse.ArgumentParser(description="AEPsych Server!")
    parser.add_argument(
        "--port", metavar="N", type=int, default=5555, help="port to serve on"
    )
    parser.add_argument(
        "--socket_type",
        choices=["zmq", "pysocket"],
        default="pysocket",
        help="method to serve over",
    )
    parser.add_argument(
        "--ip",
        metavar="M",
        type=str,
        default="0.0.0.0",
        help="ip to bind",
    )

    parser.add_argument(
        "-s",
        "--stratconfig",
        help="Location of ini config file for strat",
        type=str,
    )

    parser.add_argument(
        "--logs",
        type=str,
        help="The logs path to use if not the default (./logs).",
        default="logs",
    )

    sub_parsers = parser.add_subparsers(dest="subparser")

    database_parser = sub_parsers.add_parser("database")

    database_parser.add_argument(
        "-l",
        "--list",
        help="Lists available experiments in the database.",
        action="store_true",
    )
    database_parser.add_argument(
        "-d",
        "--db",
        type=str,
        help="The database to use if not the default (./databases/default.db).",
        default=None,
    )
    database_parser.add_argument(
        "-r", "--replay", type=str, help="UUID of the experiment to replay."
    )

    database_parser.add_argument(
        "-m", "--resume", action="store_true", help="Resume server after replay."
    )

    database_parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Update the database tables with the most recent columns.",
    )

    args = parser.parse_args()
    return args


def start_server(server_class, args):
    logger.info("Starting the AEPsychServer")
    try:
        if args.subparser == "database":
            database_path = args.db
            if args.list is True:
                database = db.Database(database_path)
                database.list_master_records()
            elif "replay" in args and args.replay is not None:
                logger.info(f"Attempting to replay {args.replay}")
                if args.resume is True:
                    sock = createSocket(socket_type=args.socket_type, port=args.port)
                    logger.info(f"Will resume {args.replay}")
                else:
                    sock = None
                startServerAndRun(
                    server_class,
                    socket=sock,
                    database_path=database_path,
                    uuid_of_replay=args.replay,
                    config_path=args.stratconfig,
                )
            elif "update" in args and args.update:
                logger.info(f"Updating the database {database_path}")
                database = db.Database(database_path)
                if database.is_update_required():
                    database.perform_updates()
                    logger.info(f"- updated database {database_path}")
                else:
                    logger.info(f"- update not needed for database {database_path}")
            else:
                logger.info(f"Setting the database path {database_path}")
                sock = createSocket(socket_type=args.socket_type, port=args.port)
                startServerAndRun(
                    server_class,
                    database_path=database_path,
                    socket=sock,
                    config_path=args.stratconfig,
                )
        else:
            sock = createSocket(socket_type=args.socket_type, port=args.port)
            startServerAndRun(server_class, socket=sock, config_path=args.stratconfig)

    except (KeyboardInterrupt, SystemExit):
        logger.exception("Got Ctrl+C, exiting!")
        sys.exit()
    except RuntimeError as e:
        fname = get_next_filename(".", "dump", "pkl")
        logger.exception(f"CRASHING!! dump in {fname}")
        raise RuntimeError(e)


def main(server_class=AEPsychServer):
    args = parse_argument()
    if args.logs:
        # overide logger path
        log_path = args.logs
        logger = utils_logging.getLogger(logging.DEBUG, log_path)
    logger.info(f"Saving logs to path: {log_path}")
    start_server(server_class, args)


if __name__ == "__main__":
    main(AEPsychServer)
