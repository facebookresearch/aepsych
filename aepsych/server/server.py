#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import json
from typing import Dict, Union, Optional
import logging
import concurrent
import traceback
import io
import os
import argparse

from aepsych.database import db
from aepsych.server.message_handlers import MESSAGE_MAP
from aepsych.server.message_handlers.handle_setup import configure
from aepsych import utils_logging, version
import numpy as np
import dill
import torch
from aepsych.server.replay import (
    get_dataframe_from_replay,
    get_strat_from_replay,
    get_strats_from_replay,
    replay,
)

logger = utils_logging.getLogger(logging.INFO)


def get_next_filename(folder, fname, ext):
    """Generates appropriate filename for logging purposes."""
    n = sum(1 for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
    return f"{folder}/{fname}_{n + 1}.{ext}"


class AEPsychServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5555,
        database_path: str = "./databases/default.db",
        max_workers: Optional[int] = None,
    ):
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.db: db.Database = db.Database(database_path)
        self.is_performing_replay = False
        self.exit_server_loop = False
        self._db_raw_record = None
        self.skip_computations = False
        self.strat_names = None
        self.extensions = None
        self._strats = []
        self._parnames = []
        self._configs = []
        self._master_records = []
        self.strat_id = -1
        self.outcome_names = []

        if self.db.is_update_required():
            self.db.perform_updates()

    #### Properties ####
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
    def _db_master_record(self):
        if self.strat_id == -1:
            return None
        else:
            return self._master_records[self.strat_id]

    @_db_master_record.setter
    def _db_master_record(self, s):
        self._master_records.append(s)

    @property
    def n_strats(self):
        return len(self._strats)

    #### Methods to handle parameter configs ####
    def _tensor_to_config(self, next_x):
        stim_per_trial = self.strat.stimuli_per_trial
        dim = self.strat.dim
        if (
            stim_per_trial > 1 and len(next_x.shape) == 2
        ):  # Multi stimuli case are complex
            # We need to find out next_x is from an ask or a query
            if stim_per_trial == next_x.shape[-1] and dim == next_x.shape[-2]:
                # From an ask, so we need to add a batch dim
                next_x = next_x.unsqueeze(0)
            # If we're a query, the 2D-ness of it is actually correct

        if len(next_x.shape) == 1:
            # We always need a batch dimension for transformations
            next_x = next_x.unsqueeze(0)

        next_x = self.strat.transforms.indices_to_str(next_x)
        config = {}
        for i, name in enumerate(self.parnames):
            val = next_x[:, i]
            if isinstance(val, str):
                config[name] = [val]
            elif isinstance(val, (int, float)):
                config[name] = [float(val)]
            elif isinstance(val[0], str):
                config[name] = val
            else:
                config[name] = list(np.array(val, dtype="float64"))
        return config

    def _config_to_tensor(self, config):
        # Converts a parameter config dictionary to a tensor
        # Check if the values of config are not array, if so make them so
        config_copy = {
            key: (
                value.squeeze()
                if isinstance(value, np.ndarray)
                else np.array(value).squeeze()
            )
            for key, value in config.items()
        }

        # Create the correctly shaped/ordered object array
        unpacked = [config_copy[name] for name in self.parnames]
        unpacked = np.stack(unpacked, axis=0, dtype="O")
        unpacked = np.expand_dims(unpacked, axis=0)  # Batch dimension,

        x = self.strat.transforms.str_to_indices(unpacked)[0]

        return x

    def _fixed_to_idx(self, fixed: Dict[str, Union[float, str]]):
        # Given a dictionary of fixed parameters, turn the parameters names into indices
        dummy = np.zeros(len(self.parnames)).astype("O")
        for key, value in fixed.items():
            idx = self.parnames.index(key)
            dummy[idx] = value
        dummy = np.expand_dims(dummy, 0)
        dummy = self.strat.transforms.str_to_indices(dummy)[0]

        # Turn the dummy back into a dict
        fixed_features = {}
        for key in fixed.keys():
            idx = self.parnames.index(key)
            fixed_features[idx] = dummy[idx].item()

        return fixed_features

    #### Methods to handle replay ####
    def replay(self, uuid_to_replay, skip_computations=False):
        return replay(self, uuid_to_replay, skip_computations)

    def get_strats_from_replay(self, uuid_of_replay=None, force_replay=False):
        return get_strats_from_replay(self, uuid_of_replay, force_replay)

    def get_strat_from_replay(self, uuid_of_replay=None, strat_id=-1):
        return get_strat_from_replay(self, uuid_of_replay, strat_id)

    def get_dataframe_from_replay(self, uuid_of_replay=None, force_replay=False):
        return get_dataframe_from_replay(self, uuid_of_replay, force_replay)

    #### Method to handle async server ####
    def start_blocking(self):
        """Starts the server in a blocking state in the main thread. Used by the
        command line interface to start the server for a client in another
        process or machine."""
        asyncio.run(self.serve())

    def start_background(self):
        """Starts the server in a background thread. Used for scripts where the
        client and server are in the same process."""
        raise NotImplementedError

    async def serve(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        self.loop = asyncio.get_running_loop()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.loop.set_default_executor(pool)

        async with server:
            logging.info(f"Serving on {self.host}:{self.port}")
            try:
                await server.serve_forever()
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

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info("peername")
        logger.info(f"Connected to {addr}")

        try:
            while True:
                rcv = await reader.read(1024 * 512)
                message = json.loads(rcv)

                future = self.loop.run_in_executor(None, self.handle_request, message)

                try:
                    result = await future
                except Exception as e:
                    logger.error(f"Error handling message: {message}")
                    logger.error(traceback.format_exc())
                    result = {"error": str(e)}

                if isinstance(result, dict):
                    return_msg = json.dumps(self._simplify_arrays(result)).encode()
                    writer.write(return_msg)
                else:
                    writer.write(str(result).encode())

                await writer.drain()
        except asyncio.CancelledError:
            pass
        finally:
            logger.info(f"Connection closed for {addr}")
            writer.close()
            await writer.wait_closed()

    def handle_request(self, message):
        type_ = message["type"]
        result = MESSAGE_MAP[type_](self, message)
        return result

    def _simplify_arrays(self, message):
        # Simplify arrays for encoding and sending a message to the client
        return {
            k: (
                v.tolist()
                if type(v) == np.ndarray
                else self._simplify_arrays(v) if type(v) is dict else v
            )
            for k, v in message.items()
        }

    #### Methods to handle exiting ####
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

    def __getstate__(self):
        # nuke the socket since it's not pickleble
        state = self.__dict__.copy()
        del state["db"]
        return state


def parse_argument():
    parser = argparse.ArgumentParser(description="AEPsych Server")
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
        default="./databases/default.db",
    )

    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        help="Unique id of the experiment to replay and resume the server from.",
    )

    args = parser.parse_args()
    return args


def main():
    logger.info("Starting AEPsychServer")
    logger.info(f"AEPsych Version: {version.__version__}")

    args = parse_argument()
    if args.logs:
        # overide logger path
        log_path = args.logs
        logger = utils_logging.getLogger(log_path)
        logger.info(f"Saving logs to path: {log_path}")

    server = AEPsychServer(
        host=args.ip,
        port=args.port,
        database_path=args.db,
    )

    if args.stratconfig is not None and args.resume is not None:
        raise ValueError(
            "Cannot configure the server with a config file and a resume from a replay at the same time."
        )

    elif args.stratconfig is not None:
        configure(server, config_str=args.stratconfig)

    elif args.resume is not None:
        if args.db is None:
            raise ValueError("Cannot resume from a replay if no database is given.")
        server.replay(args.resume, skip_computations=True)

    # Starts the server in a blocking state
    server.start_blocking()


if __name__ == "__main__":
    main()
