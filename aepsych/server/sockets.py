#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import select
import socket
import sys

import aepsych.utils_logging as utils_logging
import numpy as np

logger = utils_logging.getLogger(logging.INFO)
BAD_REQUEST = "bad request"


def SimplifyArrays(message):
    return {
        k: v.tolist()
        if type(v) == np.ndarray
        else SimplifyArrays(v)
        if type(v) is dict
        else v
        for k, v in message.items()
    }


class DummySocket(object):
    def close(self):
        pass


class PySocket(object):
    def __init__(self, port, ip=""):
        addr = (ip, port)  # all interfaces
        if socket.has_dualstack_ipv6():
            self.socket = socket.create_server(
                addr, family=socket.AF_INET6, dualstack_ipv6=True
            )
        else:
            self.socket = socket.create_server(addr)

        self.conn, self.addr = None, None

    def close(self):
        self.socket.close()

    def return_socket(self):
        return self.socket

    def accept_client(self):
        client_not_connected = True
        logger.info("Waiting for connection...")

        while client_not_connected:
            rlist, wlist, xlist = select.select([self.socket], [], [], 0)
            if rlist:
                for sock in rlist:
                    try:
                        self.conn, self.addr = sock.accept()
                        logger.info(
                            f"Connected by {self.addr}, waiting for messages..."
                        )
                        client_not_connected = False
                    except Exception as e:
                        logger.info(f"Connection to client failed with error {e}")
                        raise Exception

    def receive(self, server_exiting):
        while not server_exiting:
            rlist, wlist, xlist = select.select(
                [self.conn], [], [], 0
            )  # 0 Here is the timeout. It makes the server constantly poll for output. Timeout can be added to save CPU usage.
            # rlist,wlist,xlist represent lists of sockets to check against. Rlist is sockets to read from, wlist is sockets to write to, xlist is sockets to listen to for errors.
            for sock in rlist:
                try:
                    if rlist:
                        recv_result = sock.recv(
                            1024 * 512
                        )  # 1024 * 512 is the max size of the message
                        msg = json.loads(recv_result)
                        logger.debug(f"receive : result = {recv_result}")
                        logger.info(f"Got: {msg}")
                        return msg
                except Exception:
                    return BAD_REQUEST

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
