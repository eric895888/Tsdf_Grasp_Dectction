# Copyright 2023 tc-haung
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

import datetime
import logging
import socket
from typing import List, Union

# log_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S.log")

# logging.basicConfig(
#     level=logging.DEBUG,  # filename=log_filename, filemode='w',
#     format="%(asctime)s [%(levelname)-8s] %(message)s",
#     datefmt="%Y%m%d %H:%M:%S",
# )


class TCPClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self.socket.connect((self.host, self.port))
                logging.info(f"[TCP client] Connected to {self.host}:{self.port}")
                return True

            except OSError as msg:
                self.socket.close()
                self.socket = None
                logging.error(f"[TCP client] Connect failed: {msg}")
                return False
        except OSError as msg:
            self.socket = None
            return False

    def send(self, msg: str) -> bool:
        if self.socket is None:
            logging.error(f"[TCP client] Socket is None")
            return False
        else:
            self.socket.send(msg.encode())
            logging.info(f"[TCP client] Sent: {msg}")
            return True

    def recv(self, size=1024) -> Union[List[str], None]:
        try:
            response = self.socket.recv(size).decode()
            logging.info(f"[TCP client] Received: {response}")

        except OSError as msg:
            logging.error(f"[TCP client] Receive failed: {msg}")
            return None

        if response.find(";") == -1:
            logging.error(f"[TCP client] Response error: {response} not include ';'")
            # TODO : Check return None or response (original)
            return None

        # TODO : Unknow motivation
        if response.find(";") != len(response) - 1:
            response = response[0 : response.find(";") + 1]

        # TODO: Check this
        # split_response = response.split(',') # modified
        split_response = response.split(",", 5)  # original

        if len(split_response) != 6:
            split_response[0] = split_response[0].split(";", 1)[0]
        else:
            split_response[5] = split_response[5].split(";", 1)[0]

        logging.info(f"[TCP client] Splited response: {split_response}")

        return split_response

    def close(self) -> bool:
        if self.socket is None:
            logging.warning(f"[TCP client] Close socket failed: Socket is None")
            return False
        else:
            self.socket.close()
            logging.info(f"[TCP client] Closed socket")
            return True

    # def is_connected(self) -> bool:
    #     if self.socket is None:
    #         return False
    #     else:
    #         return True

    # def log(self):
    #     logging.debug('debug')
    #     logging.info('info')
    #     logging.warning('warning')
    #     logging.error('error')
    #     logging.critical('critical')
