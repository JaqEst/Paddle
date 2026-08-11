# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import socket


logger = logging.getLogger(__name__)


class ControlSocketServer:
    def __init__(self, path: str, backlog=8, max_message_size=4 * 1024):
        self.path = path
        self.backlog = backlog
        self.max_message_size = max_message_size
        self.sock = None

    def start(self):
        if not self.path:
            return

        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

        parent_dir = os.path.dirname(self.path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.path)
        self.sock.listen(self.backlog)
        self.sock.setblocking(False)
        logger.info(f"Start launch control socket server at {self.path}")

    def poll(self, handler):
        if self.sock is None:
            return

        while True:
            try:
                conn, _ = self.sock.accept()
            except BlockingIOError:
                return

            with conn:
                result = None
                try:
                    command = self._recv_json(conn)
                    result = handler(command)
                except Exception as e:
                    logger.warning(f"Handle launch control command failed: {e}")
                    result = {"status": "error", "message": str(e)}

                conn.sendall(json.dumps(result).encode("utf-8"))

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

        if self.path:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass

    def _recv_json(self, conn):
        chunks = []
        size = 0
        while True:
            chunk = conn.recv(1024)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > self.max_message_size:
                raise ValueError(
                    f"control socket message is too large: {size} > {self.max_message_size}"
                )

        if not chunks:
            raise ValueError("empty control socket message")

        return json.loads(b"".join(chunks).decode("utf-8"))


class ControlSocketClient:
    def __init__(self, path: str, timeout=5.0, max_message_size=4 * 1024):
        self.path = path
        self.timeout = timeout
        self.max_message_size = max_message_size

    def request(self, command):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect(self.path)
            sock.sendall(json.dumps(command).encode("utf-8"))
            sock.shutdown(socket.SHUT_WR)
            return self._recv_json(sock)

    def _recv_json(self, sock):
        chunks = []
        size = 0
        while True:
            chunk = sock.recv(1024)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > self.max_message_size:
                raise ValueError(
                    f"control socket response is too large: {size} > {self.max_message_size}"
                )

        if not chunks:
            raise ValueError("empty control socket response")

        return json.loads(b"".join(chunks).decode("utf-8"))
