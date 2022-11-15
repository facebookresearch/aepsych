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
import zmq

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


def createSocket(socket_type="pysocket", port=5555, msg_queue=None):
    logger.info(f"socket_type = {socket_type} port = {port}")

    if socket_type == "pysocket":
        sock = PySocket(port=port)
    elif socket_type == "zmq":
        sock = ZMQSocket(port=port)
    elif socket_type == "thrift":
        sock = ThriftSocketWrapper(msg_queue)

    return sock


class DummySocket(object):
    def close(self):
        pass


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
        while True:
            try:
                msg = self.socket.recv_json()
                break
            except Exception as e:
                logger.info(
                    "Exception caught while trying to receive a message from the client. "
                    f"Ignoring message and trying again. The caught exception was: {e}."
                )
        return msg

    def send(self, message):
        if type(message) == str:
            self.socket.send_string(message)
        elif type(message) == int:
            self.socket.send_string(str(message))
        else:
            self.socket.send_json(SimplifyArrays(message))


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
                except Exception as e:
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


class ThriftSocketWrapper(object):
    def __init__(self, msg_queue=None):
        self.msg_queue = msg_queue

    def close(self):
        # it's not a real socket so no close function need?
        pass

    def receive(self):
        # Remove and return an item from the queue. If queue is empty, wait until an item is available.
        message = self.msg_queue.get()
        logger.info(f"thrift socket got msg: {message}")
        return message

    def send(self, message):
        # add responds to msg_queue
        if self.msg_queue is None:
            logger.exception("There is no msg_queue!")
            raise RuntimeError("There is no message to send from server!")
        if type(message) == str:
            pass  # keep it as-is
        elif type(message) == int:
            message = str(message)
        else:
            message = json.dumps(SimplifyArrays(message))
        logger.info(f"Sending: {message}")
        self.msg_queue.put(message, block=True)
