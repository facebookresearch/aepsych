#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import socket
from typing import Any, Dict, List


class AEPsychClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5555, connect=True) -> None:
        """Python client for AEPsych using built-in python sockets. By default it connects
        to a localhost server matching AEPsych defaults.

        Args:
            ip (str): IP to connect to (default: localhost).
            port (str): Port to connect on (default: 5555).
            connect (bool, optional): Connect as part of init? Defaults to True.
        """

        self.socket = socket.socket()
        self.configs = []
        self.config_names = {}
        if connect:
            self.connect(ip, port)

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
        message = bytes(json.dumps(message), encoding="utf-8")
        self.socket.send(message)
        response = self.socket.recv(4096).decode("utf-8")
        if response == "bad request":
            raise RuntimeError(f"Bad request '{message}'")

        return response

    def ask(self) -> Dict[str, Any]:
        """Get next configuration from server.

        Returns:
            Dict[str, str]: Next configuration to evaluate.
        """
        request = {"message": "", "type": "ask", "version": "0.01"}
        response = json.loads(self._send_recv(request))
        return response

    def tell(
        self, config: Dict[str, List[Any]], outcome: int, **metadata: Dict[str, Any]
    ) -> None:
        """Update the server on a configuration that was executed.

        Args:
            config (Dict[str, str]): Config that was evaluated.
            outcome (int): Outcome that was obtained.
            metadata (optional kwargs) is passed to the extra_info field on the server.

        Raises:
            AssertionError if server failed to acknowledge the tell.
        """

        request = {
            "type": "tell",
            "message": {"config": config, "outcome": outcome},
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
