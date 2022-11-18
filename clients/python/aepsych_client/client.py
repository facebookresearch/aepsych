#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import socket
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aepsych.server import AEPsychServer

class ServerError(RuntimeError):
    pass

class AEPsychClient:
    def __init__(
        self,
        ip: Optional[str] = None,
        port: Optional[int] = None,
        connect: bool = True,
        server: "AEPsychServer" = None,
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
        self.configs = []
        self.config_names = {}
        self.server = server

        if server is not None and (ip is not None or port is not None):
            warnings.warn(
                "AEPsychClient will ignore ip and port since it was given a server object!",
                UserWarning,
            )

        if server is None:
            ip = ip or "0.0.0.0"
            port = port or 5555

            self.socket = socket.socket()
            if connect:
                self.connect(ip, port)

    def load_config_index(self) -> None:
        """Loads the config index when server is not None"""
        self.configs = []
        for i in range(self.server.n_strats):
                    self.configs.append(i)

    def connect(self, ip: str, port: int) -> None:
        """Connect to the server.

        Args:
            ip (str): IP to connect to.
            port (str): Port to connect on.
        """
        addr = (ip, port)
        self.socket.connect(addr)

    def finalize(self) -> None:
        """Let the server know experiment is complete."""
        request = {"message": "", "type": "exit"}
        self._send_recv(request)

    def _send_recv(self, message) -> str:
        if self.server is not None:
            return self.server.handle_request(message)

        message = bytes(json.dumps(message), encoding="utf-8")
        self.socket.send(message)
        response = self.socket.recv(4096).decode("utf-8")
        # TODO this is hacky but we don't consistencly return json
        # from the server so we can't check for a status
        if response[:12] == "server_error":
            error_message = response[13:]
            raise ServerError(error_message)

        return response

    def ask(self) -> Dict[str, Any]:
        """Get next configuration from server.

        Returns:
            Dict[str, str]: Next configuration to evaluate.
        """
        request = {"message": "", "type": "ask", "version": "0.01"}
        response = self._send_recv(request)
        if isinstance(response, str):
            response = json.loads(response)
        return response

    def tell(
        self,
        config: Dict[str, List[Any]],
        outcome: int,
        model_data: bool = True,
        **metadata: Dict[str, Any],
    ) -> None:
        """Update the server on a configuration that was executed.

        Args:
            config (Dict[str, str]): Config that was evaluated.
            outcome (int): Outcome that was obtained.
            metadata (optional kwargs) is passed to the extra_info field on the server.
            model_data (bool): If True, the data will be recorded in the db and included in the server's model. If False,
                the data will be recorded in the db, but will not be used by the model. Defaults to True.

        Raises:
            AssertionError if server failed to acknowledge the tell.
        """

        request = {
            "type": "tell",
            "message": {"config": config, "outcome": outcome, "model_data": model_data},
            "extra_info": metadata,
        }
        self._send_recv(request)

    def configure(
        self, config_path: str = None, config_str: str = None, config_name: str = None
    ) -> None:
        """Configure the server and prepare for data collection.
        Note that either config_path or config_str must be passed.

        Args:
            config_path (str, optional): Path to a config.ini. Defaults to None.
            config_str (str, optional): Config.ini encoded as a string. Defaults to None.
            config_name (str, optional): A name to assign to this config internally for convenience.

        Raises:
            AssertionError if neither config path nor config_str is passed.
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
            "version": "0.01",
            "message": {"config_str": config_str},
        }
        idx = int(self._send_recv(request))
        self.configs.append(idx)
        if config_name is not None:
            self.config_names[config_name] = idx

    def resume(self, config_id: int = None, config_name: str = None):
        """Resume a previous config from this session. To access available configs,
        use client.configs or client.config_names

        Args:
            config_id (int, optional): ID of config to resume.
            config_name (str, optional): Name config to resume.

        Raises:
            AssertionError if name or ID does not exist, or if both name and ID are passed.
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
            "version": "0.01",
            "message": {"strat_id": config_id},
        }
        self._send_recv(request)

    def __del___(self):
        self.finalize()
