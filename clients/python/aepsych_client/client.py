#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import socket
import warnings
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch
    from aepsych.server import AEPsychServer


class ServerError(RuntimeError):
    pass


class AEPsychClient:
    def __init__(
        self,
        ip: Optional[str] = None,
        port: Optional[int] = None,
        connect: bool = True,
        server: Optional["AEPsychServer"] = None,
    ) -> None:
        """Python client for AEPsych using built-in python sockets. By default it connects
        to a localhost server matching AEPsych defaults.

        Args:
            ip (str, optional): IP to connect to (default: localhost).
            port (str, optional): Port to connect on (default: 5555).
            connect (bool): Connect as part of init? Defaults to True.
            server (AEPsychServer, optional): An in-memory AEPsychServer object to connect to.
                If this is not None, the other arguments will be ignored.
        """
        self.configs: List[int] = []
        self.config_names: Dict[str, int] = {}
        self.server = server

        if server is not None and (ip is not None or port is not None):
            warnings.warn(
                "AEPsychClient will ignore ip and port since it was given a server object!",
                UserWarning,
            )

        if server is None:
            ip = ip or "0.0.0.0"
            port = port or 5555

            self.socket: Optional[socket.socket] = socket.socket()
            if connect:
                self.connect(ip, port)
        else:
            self.socket = None

    def load_config_index(self) -> None:
        """Loads the config index when there is an in-memory server."""
        if self.server is None:
            raise AttributeError("there is no in-memory server")

        self.configs = []
        for i in range(self.server.n_strats):
            self.configs.append(i)

    def connect(self, ip: str, port: int) -> None:
        """Connect to the server.

        Args:
            ip (str): IP to connect to.
            port (str): Port to connect on.
        """
        if self.socket is None:
            raise AttributeError("client does not have a socket to connect with")

        addr = (ip, port)
        self.socket.connect(addr)

    def finalize(self) -> Dict[str, Any]:
        """Let the server know experiment is complete and stop the server.

        Returns:
            Dict[str, Any]: A dictionary with two entries:
            - "config": dictionary with config (keys are strings, values are floats).
                Currently always "Terminate" if this function succeeds.
            - "is_finished": boolean, true if the strat is finished. Currently always
                true if this function succeeds.
        """
        request = {"message": "", "type": "exit"}
        return self._send_recv(request)

    def _send_recv(self, message) -> Dict[str, Any]:
        # Sends a message to a server and decodes the response
        if self.server is not None:
            return self.server.handle_request(message)

        if self.socket is None:
            raise AttributeError("client does not have a socket to connect with")

        message = bytes(json.dumps(message), encoding="utf-8")
        self.socket.send(message)
        response = self.socket.recv(4096).decode("utf-8")

        response_dict = json.loads(response)

        if "server_error" in response_dict:
            raise ServerError(response_dict["server_error"])

        return response_dict

    def ask(
        self, num_points: int = 1
    ) -> Union[Dict[str, List[float]], Dict[int, Dict[str, Any]]]:
        """Get next configuration from server.

        Args:
            num_points[int]: Number of points to return.

        Returns:
            Dict[str, Any]: A dictionary with three entries
                - "config": dictionary with config (keys are strings, values are floats), None
                    if skipping computations during replay.
                - "is_finished": boolean, true if the strat is finished
                - "num_points": integer, number of points returned.
        """
        request = {"message": {"num_points": num_points}, "type": "ask"}
        response = self._send_recv(request)

        return response

    def tell(
        self,
        config: Dict[str, List[Any]],
        outcome: Union[float, Dict[str, float]],
        model_data: bool = True,
        **metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update the server on a configuration that was executed. Use this method when using the legacy backend or for
        manually-generated trials without an associated trial_index when uding the Ax backend.

        Args:
            config (Dict[str, str]): Config that was evaluated.
            outcome (int): Outcome that was obtained.
            metadata (optional kwargs) is passed to the extra_data field on the server.
            model_data (bool): If True, the data will be recorded in the db and included in the server's model. If False,
                the data will be recorded in the db, but will not be used by the model. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary with these entries
                - "trials_recorded": integer, the number of trials recorded in the
                    database.
                - "model_data_added": integer, the number of datapoints added to the model.
        """
        message = {
            "config": config,
            "outcome": outcome,
            "model_data": model_data,
        }
        message.update(**metadata)
        request = {
            "type": "tell",
            "message": message,
        }

        return self._send_recv(request)

    def configure(
        self,
        config_path: Optional[str] = None,
        config_str: Optional[str] = None,
        config_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Configure the server and prepare for data collection.
        Note that either config_path or config_str must be passed.

        Args:
            config_path (str, optional): Path to a config.ini. Defaults to None.
            config_str (str, optional): Config.ini encoded as a string. Defaults to None.
            config_name (str, optional): A name to assign to this config internally for convenience.

        Returns:
            Dict[str, Any]: A dictionary with one entry
                - "strat_id": integer, the stategy ID for what was just set up.
        """

        if config_path is not None:
            assert config_str is None, "if config_path is passed, don't pass config_str"
            with open(config_path, "r") as f:
                config_str = f.read()
        elif config_str is not None:
            assert (
                config_path is None
            ), "if config_str is passed, don't pass config_path"
        request = {
            "type": "setup",
            "message": {"config_str": config_str},
        }
        response = self._send_recv(request)
        self.configs.append(response["strat_id"])
        if config_name is not None:
            self.config_names[config_name] = response["strat_id"]

        return response

    def resume(
        self, config_id: Optional[int] = None, config_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Resume a previous config from this session. To access available configs,
        use client.configs or client.config_names

        Args:
            config_id (int, optional): ID of config to resume.
            config_name (str, optional): Name config to resume.

        Returns:
            Dict[str, Any]: A dictionary with one entry
                - "strat_id": integer, the stategy ID that was resumed.
        """
        if config_id is not None:
            assert config_name is None, "if config_id is passed, don't pass config_name"
            assert (
                config_id in self.configs
            ), f"No strat with index {config_id} was created!"
        elif config_name is not None:
            assert config_id is None, "if config_name is passed, don't pass config_id"
            assert (
                config_name in self.config_names.keys()
            ), f"{config_name} not known, know {self.config_names.keys()}!"
            config_id = self.config_names[config_name]
        request = {
            "type": "resume",
            "message": {"strat_id": config_id},
        }

        return self._send_recv(request)

    def query(
        self,
        query_type: Literal["max", "min", "prediction", "inverse"] = "max",
        probability_space: bool = False,
        x: Optional[Dict[str, Any]] = None,
        y: Optional[Union[float, "torch.Tensor"]] = None,
        constraints: Optional[Dict[int, float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Queries the underlying model for a specific query.

        Args:
            query_type (Literal["max", "min", "prediction", "inverse"]): What type of query
                to make. Defaults to "max".
            probability_space (bool): Whether the y in the query is in probability space.
                Defaults to False.
            x (Dict[str, Any], optional): A parameter configuration dictionary representing
                one or more point for a prediction query.
            y (Union[float, torch.Tensor], optional): The expected y for a inverse query.
            constraints (Dict[int, float], optional): The constraints to impose on the
                query where each key is the parameter index and the value is the parameter
                value to apply the equality constraint at.
            **kwargs: Additional kwargs to pass to the query function.

        Returns:
            Dict[str, Any: A dictionary with these entries:
                - "query_response": string, the query response.
                - "probability_space": boolean, whether to query in the probability space
                    or not.
                - "constraints": dictionary, the equality constraint for parameters
                    where the keys are the parameter index and the values are the point
                    where the paramter should be constrained to.
                - "x": dictionary, the parameter configuration dictionary for the query.
                - "y": list, the y from the query.
        """
        message = {
            "query_type": query_type,
            "probability_space": probability_space,
            "x": x,
            "y": y,
            "constraints": constraints,
        }
        message.update(**kwargs)

        request = {"type": "query", "message": message}

        return self._send_recv(request)

    def __del__(self):
        self.finalize()
