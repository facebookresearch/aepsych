#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import io
import json
import logging
import os
import socket
import sys
import warnings

import aepsych.database.db as db
import aepsych.utils_logging as utils_logging
import dill
import numpy as np
import pandas as pd
import torch
import zmq
from aepsych.config import Config
from aepsych.strategy import SequentialStrategy


logger = utils_logging.getLogger(logging.DEBUG)


def SimplifyArrays(message):
    return {k: v.tolist() if type(v) == np.ndarray else v for k, v in message.items()}


def _get_next_filename(folder, fname, ext):
    n = sum(1 for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
    return f"{folder}/{fname}_{n+1}.{ext}"


class ZMQSocket(object):
    def __init__(self, port, ip="*"):
        """sends/receives json-formated messages over ZMQ

        Arguments:
            port {int} -- port to listen over
        """

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{ip}:{port}")

    def close(self):
        self.socket.close()

    def receive(self):
        return self.socket.recv_json()

    def send(self, message):
        if type(message) == str:
            self.socket.send_string(message)
        elif type(message) == int:
            self.socket.send_string(str(message))
        else:
            self.socket.send_json(SimplifyArrays(message))


class PySocket(object):
    def __init__(self, port, ip="0.0.0.0"):

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((ip, port))
        self.socket.listen()
        self.conn = None

    def close(self):
        self.socket.close()

    def receive(self):
        if self.conn is None:
            logger.info("Waiting for connection...")
            self.conn, self.addr = self.socket.accept()
        recv_result = b""
        while recv_result == b"":
            logger.info(f"Connected by {self.addr}, waiting for messages...")
            recv_result = self.conn.recv(1024 * 512)  # 512KiB
            logger.debug(f"receive : result = {recv_result}")
            msg = json.loads(recv_result)

        logger.info(f"Got: {msg}")
        return msg

    def send(self, message):
        if self.conn is None:
            logger.error("No connection to send to!")
            return
        if type(message) == str:
            pass  # keep it as-is
        elif type(message) == int:
            message = str(message)
        else:
            message = json.dumps(SimplifyArrays(message))
        logger.info(f"Sending: {message}")
        sys.stdout.flush()
        self.conn.sendall(bytes(message, "utf-8"))


class AEPsychServer(object):
    def __init__(self, socket=None, database_path=None):
        """Server for doing black box optimization using gaussian processes.

        Keyword Arguments:
            socket -- socket object that implements `send` and `receive` for json
            messages (default: ZMQSocket).
            TODO actually make an abstract interface to subclass from here
        """
        if socket is None:
            self.socket = PySocket(port=5555)
        else:
            self.socket = socket

        self.db = None
        self.is_performing_replay = False
        self.exit_server_loop = False
        self._db_master_record = None
        self.db = db.Database(database_path)

        if self.db.is_update_required():
            raise RuntimeError(
                f'The database needs to be updated. You can perform the update by running "python3 aepsych/server.py database --update --d {database_path}"'
            )

        self._strats = []
        self.strat_id = -1

        self.debug = False

    def cleanup(self):
        self.socket.close()

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
        while True:
            request = self.socket.receive()

            if "version" in request.keys():
                result = self.versioned_handler(request)
            else:
                result = self.unversioned_handler(request)

            self.socket.send(result)
            if self.exit_server_loop:
                break

    def replay(self, uuid_to_replay, skip_asks=False):
        """
        Run a replay against the server. The UUID will be looked up in the database.
        if skip_asks is true, skip all the asks, which should make the replay much faster.
        """
        if uuid_to_replay is None:
            raise RuntimeError("UUID is a required parameter to perform a replay")

        if self.db is None:
            raise RuntimeError("A database is required to perform a replay")

        if skip_asks is True:
            logger.info(
                "skip_asks=True, make sure to refit the final strat before doing anything!"
            )

        master_record = self.db.get_master_record(uuid_to_replay)

        if master_record is None:
            raise RuntimeError(
                f"The UUID {uuid_to_replay} isn't in the database. Unable to perform replay."
            )

        # this prevents writing back to the DB and creating a circular firing squad
        self.is_performing_replay = True

        # if there is a config in the DB we'll use it.
        if 0 < len(master_record.children_config):
            self._configure(master_record.children_config[0].config)

        for result in master_record.children_replay:
            request = result.message_contents
            logger.debug(f"replay - type = {result.message_type} request = {request}")
            if request["type"] == "ask" and skip_asks is True:
                logger.debug("Request type is ask and skip_asks==True, skipping!")
                # HACK increment strat's count and manually move to next strat as needed, since
                # strats count based on `gen` calls not `add_data calls`.
                # TODO this should probably be the other way around when we refactor
                # the whole Modelbridge/Strategy axis.
                self.strat._strat._count += 1
                if isinstance(self.strat, SequentialStrategy) and self.strat._count >= self.strat._strat.n_trials:
                    self.strat._make_next_strat()
                continue
            if "version" in request.keys():
                result = self.versioned_handler(request)
            else:
                result = self.unversioned_handler(request)

    def get_final_strat_from_replay(self, uuid_of_replay=None):
        if uuid_of_replay is None:
            records = self.db.get_master_records()
            if len(records) > 0:
                uuid_of_replay = records[-1].experiment_id
            else:
                raise RuntimeError("Server has no experiment records!")
        try:
            strat_buffer = self.db.get_strat_for(uuid_of_replay)
            return torch.load(strat_buffer, pickle_module=dill)
        except Exception as e:
            logger.info("No final strat found (likely due to old DB or server crash, "+
                        "trying to replay tells to generate a final strat...")
            try:
                # sometimes there's no final strat, e.g.
                # if the server crashed or it's a very old database
                # in this case, replay the setup and tells
                self.replay(uuid_of_replay, skip_asks=True)
                # then if the final strat is model-based, refit
                if self.strat.has_model:
                    self.strat.modelbridge.fit(self.strat.x, self.strat.y)
                return self.strat
            except Exception:
                raise e

    def _flatten_tell_record(self, rec):
        out = {}
        out["response"] = int(rec.message_contents["message"]["outcome"])
        out.update(
            {k: v[0] for k, v in rec.message_contents["message"]["config"].items()}
        )

        if rec.extra_info is not None:
            out.update(rec.extra_info)
        return out

    def get_dataframe_from_replay(self, uuid_of_replay=None):
        if uuid_of_replay is None:
            uuid_of_replay = self.db.get_master_records()[-1].experiment_id
        recs = self.db.get_replay_for(uuid_of_replay)

        strat = self.get_final_strat_from_replay(uuid_of_replay)

        post_mean, post_var = strat.predict(strat.x)

        out = pd.DataFrame(
            [
                self._flatten_tell_record(rec)
                for rec in recs
                if rec.message_type == "tell"
            ]
        )
        out["post_mean"] = post_mean.detach().numpy()
        out["post_var"] = post_var.detach().numpy()
        return out

    def versioned_handler(self, request):
        handled_types = ["setup", "resume"]
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
        if request["type"] in handled_types:
            logger.debug(f"Received msg [{request['type']}]")

        else:
            warnings.warn(
                "Got versioned handler but no version, falling back to unversioned!"
            )
            self.unversioned_handler(request)
        return ret_val

    def handle_setup_v01(self, request):
        logger.debug("got setup message!")

        if not self.is_performing_replay:
            experiment_id = None
            if self._db_master_record is not None:
                experiment_id = self._db_master_record.experiment_id

            self._db_master_record = self.db.record_setup(
                description="default description",
                name="default name",
                request=request,
                id=experiment_id,
            )

        if (
            "config_str" in request["message"].keys()
            or "config_dict" in request["message"].keys()
        ):
            strat_id = self.configure(**request["message"])
        else:
            strat_id = self.setup(**request["message"])

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

    def unversioned_handler(self, request):
        message_map = {
            "setup": self.handle_setup,
            "ask": self.handle_ask,
            "tell": self.handle_tell,
            "update": self.handle_update,
        }

        if "type" not in request.keys():
            raise RuntimeError(f"Request {request} contains no request type!")
        else:
            type = request["type"]
            if type in message_map.keys():
                logger.debug(f"Received msg [{type}]")
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
                description="default description",
                name="default name",
                request=request,
                id=experiment_id,
            )

        self.setup(**request["message"])
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
    def n_strats(self):
        return len(self._strats)

    def ask(self):
        """get the next point to query from the model

        Returns:
            dict -- new config dictionary (keys are strings, values are floats)
        """
        # index by [0] is temporary HACK while serverside
        # doesn't handle batched ask
        next_x = self.strat.gen()[0]

        return self._tensor_to_config(next_x)

    def _tensor_to_config(self, next_x):
        config = {}
        for name, val in zip(self.parnames, next_x):
            config[name] = [float(val)]
        return config

    def _config_to_tensor(self, config):
        x = np.stack([config[name] for name in self.parnames], axis=1)

        return x

    def tell(self, outcome, config):
        """tell the model which input was run and what the outcome was

        Arguments:
            inputs {dict} -- dictionary, keys are strings, values are floats or int.
            keys should inclde all of the parameters we are tuning over, plus 'outcome'
            which would be in {0, 1}.
            TODO better types
        """

        x = self._config_to_tensor(config)
        y = outcome
        self.strat.add_data(x, y)

    def _configure(self, config):
        self.parnames = config.str_to_list(
            config.get("experiment", "parnames"), element_type=str
        )
        self.outcome_type = config.get(
            "common", "outcome_type", fallback="single_probit"
        )

        self.strat = SequentialStrategy.from_config(config)
        self.strat_id = self.n_strats - 1  # 0-index strats
        return self.strat_id

    def configure(self, **config_args):
        config = Config(**config_args)
        return self._configure(config)

    def __getstate__(self):
        ### nuke the socket since it's not pickleble
        state = self.__dict__.copy()
        del state["socket"]
        del state["db"]
        return state


def startServerAndRun(
    server_class, socket=None, database_path=None, config_path=None, uuid_of_replay=None
):
    try:
        server = server_class(socket=socket, database_path=database_path)
        if config_path is not None:
            with open(config_path) as f:
                config_str = f.read()
            server.configure(config_str)

        if socket is not None:
            server.serve()
        else:
            if config_path is not None:
                logger.info(
                    "You have passed in a config path but this is a replay. If there's a config in the database it will be used instead of the passed in config path."
                )
            server.replay(uuid_of_replay)
    except (KeyboardInterrupt, SystemExit):
        if server._db_master_record is not None and server.strat is not None:
            logger.info("Dumping strat to DB due to CTRL+C.")
            buffer = dill.dumps(server.strat)
            server.db.record_strat(master_table=server._db_master_record, strat=buffer)

        fname = _get_next_filename(".", "dump", "pkl")

        logger.exception(f"Got Ctrl+C, exiting! server dump in {fname}")
        dill.dump(server, open(fname, "wb"))
    except RuntimeError as e:
        if server._db_master_record is not None and server.strat is not None:
            logger.info("Dumping strat to DB due to exception.")
            buffer = dill.dumps(server.strat)
            server.db.record_strat(master_table=server._db_master_record, strat=buffer)

        fname = _get_next_filename(".", "crashdump", "pkl")

        logger.exception(f"CRASHING!! dump in {fname}")
        dill.dump(server, open(fname, "wb"))
        raise RuntimeError(e)


def createSocket(socket_type="pysocket", port=5555):
    logger.info(f"socket_type = {socket_type} port = {port}")

    if socket_type == "pysocket":
        sock = PySocket(port=port)
    elif socket_type == "zmq":
        sock = ZMQSocket(port=port)

    return sock

def start_server(server_class):
    logger.info("Starting the AEPsychServer")
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
        "-u",
        "--update",
        action="store_true",
        help="Update the database tables with the most recent columns.",
    )

    args = parser.parse_args()

    try:
        if args.subparser == "database":
            database_path = args.db
            if args.list is True:
                database = db.Database(database_path)
                database.list_master_records()
            elif "replay" in args and args.replay is not None:
                logger.info(f"Attempting to replay {args.replay}")
                startServerAndRun(
                    server_class,
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
                parser.print_help()
        else:
            sock = createSocket(socket_type=args.socket_type, port=args.port)
            startServerAndRun(server_class, socket=sock, config_path=args.stratconfig)

    except (KeyboardInterrupt, SystemExit):
        logger.exception("Got Ctrl+C, exiting!")
        sys.exit()
    except RuntimeError as e:
        fname = _get_next_filename(".", "dump", "pkl")
        logger.exception(f"CRASHING!! dump in {fname}")
        raise RuntimeError(e)


if __name__ == "__main__":
    start_server(AEPsychServer)
