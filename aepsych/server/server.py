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

from typing import Optional

import aepsych.database.db as db

import aepsych.utils_logging as utils_logging
import dill
import numpy as np
import pandas as pd
import torch

from aepsych.server.message_handlers import MESSAGE_MAP
from aepsych.server.message_handlers.handle_ask import ask
from aepsych.server.message_handlers.handle_setup import configure
from aepsych.server.replay import (
    get_dataframe_from_replay,
    get_strat_from_replay,
    get_strats_from_replay,
    replay,
)

from aepsych.server.sockets import BAD_REQUEST, DummySocket, PySocket

logger = utils_logging.getLogger(logging.INFO)
DEFAULT_DESC = "default description"
DEFAULT_NAME = "default name"


def get_next_filename(folder, fname, ext):
    """Generates appropriate filename for logging purposes."""
    n = sum(1 for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
    return f"{folder}/{fname}_{n+1}.{ext}"


class AEPsychServer(object):
    def __init__(self, socket=None, database_path=None):
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
        self.strat_names = None

        if self.db.is_update_required():
            self.db.perform_updates()

        self._strats = []
        self._parnames = []
        self._configs = []
        self.strat_id = -1
        self._pregen_asks = []
        self.enable_pregen = False
        self.outcome_names = []

        self.debug = False
        self.receive_thread = threading.Thread(
            target=self._receive_send, args=(self.exit_server_loop,), daemon=True
        )

        self.queue = []

    def cleanup(self):
        """Close the socket and terminate connection to the server.

        Returns:
            None
        """
        self.socket.close()

    def _receive_send(self, is_exiting: bool) -> None:
        """Receive messages from the client.

        Args:
            is_exiting (bool): True to terminate reception of new messages from the client, False otherwise.

        Returns:
            None
        """
        while True:
            request = self.socket.receive(is_exiting)
            if request != BAD_REQUEST:
                self.queue.append(request)
            if self.exit_server_loop:
                break
        logger.info("Terminated input thread")

    def _handle_queue(self) -> None:
        """Handles the queue of messages received by the server.

        Returns:
            None
        """
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
        else:
            if self.can_pregen_ask and (len(self._pregen_asks) == 0):
                self._pregen_asks.append(ask(self))

    def serve(self) -> None:
        """Run the server. Note that all configuration outside of socket type and port
        happens via messages from the client. The server simply forwards messages from
        the client to its `setup`, `ask` and `tell` methods, and responds with either
        acknowledgment or other response as needed. To understand the server API, see
        the docs on the methods in this class.

        Returns:
            None

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
        self.socket.accept_client()
        self.receive_thread.start()
        while True:
            self._handle_queue()
            if self.exit_server_loop:
                break
        # Close the socket and terminate with code 0
        self.cleanup()
        sys.exit(0)

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

    def generate_experiment_table(
        self,
        experiment_id: str,
        table_name: str = "experiment_table",
        return_df: bool = False,
    ) -> Optional[pd.DataFrame]:
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
        else:
            return None

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

    @property
    def can_pregen_ask(self):
        return self.strat is not None and self.enable_pregen

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
        if "type" not in request.keys():
            raise RuntimeError(f"Request {request} contains no request type!")
        else:
            type = request["type"]
            if type in MESSAGE_MAP.keys():
                logger.info(f"Received msg [{type}]")
                ret_val = MESSAGE_MAP[type](self, request)
                return ret_val

            else:
                exception_message = (
                    f"unknown type: {type}. Allowed types [{MESSAGE_MAP.keys()}]"
                )

                raise RuntimeError(exception_message)

    def replay(self, uuid_to_replay, skip_computations=False):
        return replay(self, uuid_to_replay, skip_computations)

    def get_strats_from_replay(self, uuid_of_replay=None, force_replay=False):
        return get_strats_from_replay(self, uuid_of_replay, force_replay)

    def get_strat_from_replay(self, uuid_of_replay=None, strat_id=-1):
        return get_strat_from_replay(self, uuid_of_replay, strat_id)

    def get_dataframe_from_replay(self, uuid_of_replay=None, force_replay=False):
        return get_dataframe_from_replay(self, uuid_of_replay, force_replay)


#! THIS IS WHAT START THE SERVER
def startServerAndRun(
    server_class, socket=None, database_path=None, config_path=None, uuid_of_replay=None
):
    server = server_class(socket=socket, database_path=database_path)
    try:
        if config_path is not None:
            with open(config_path) as f:
                config_str = f.read()
            configure(server, config_str=config_str)

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
    except KeyboardInterrupt:
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

    parser.add_argument(
        "-d",
        "--db",
        type=str,
        help="The database to use if not the default (./databases/default.db).",
        default=None,
    )
    parser.add_argument(
        "-r", "--replay", type=str, help="UUID of the experiment to replay."
    )

    parser.add_argument(
        "-m", "--resume", action="store_true", help="Resume server after replay."
    )

    args = parser.parse_args()
    return args


def start_server(server_class, args):
    logger.info("Starting the AEPsychServer")
    try:
        if "db" in args and args.db is not None:
            database_path = args.db
            if "replay" in args and args.replay is not None:
                logger.info(f"Attempting to replay {args.replay}")
                if args.resume is True:
                    sock = PySocket(port=args.port)
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
            else:
                logger.info(f"Setting the database path {database_path}")
                sock = PySocket(port=args.port)
                startServerAndRun(
                    server_class,
                    database_path=database_path,
                    socket=sock,
                    config_path=args.stratconfig,
                )
        else:
            sock = PySocket(port=args.port)
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
