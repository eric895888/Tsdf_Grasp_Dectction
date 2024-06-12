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

import binascii
import logging
import time

import serial

PORT = "/dev/ttyUSB0"
BAUDRATE = 115200
TIMEOUT = 0.1


class Gripper2F85:
    def __init__(self, parent=None):
        self.ser = None

    def connect(self) -> bool:
        try:
            logging.info(
                f"[2F85] Connected to gripper: port={PORT}, baudrate={BAUDRATE}, timeout={TIMEOUT}"
            )
            self.ser = serial.Serial(
                port=PORT,
                baudrate=BAUDRATE,
                timeout=TIMEOUT,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
            )
            return True

        except serial.serialutil.SerialException as e:
            logging.error(f"[2F85] Connect failed: {e}")
            return False

    def disconnect(self) -> bool:
        try:
            self.ser.close()
            logging.info(f"[2F85] Disconnected from gripper successfully")
            return True
        except serial.serialutil.SerialException as e:
            logging.error(f"[2F85] Disconnect failed: {e}")
            return False

    def reset(self) -> bool:
        # Activate gripper
        logging.info(f"[2F85] Activate gripper")
        try:
            logging.debug(f"[2F85] Clear: 09 10 03 E8 00 03 06 00 00 00 00 00 00 73 30")
            self.ser.write(
                b"\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30"
            )
            data_raw = self.ser.readline()  # bytes
            data = binascii.hexlify(data_raw)
            logging.info(f"[2F85] Activation's Response: {data.decode()}")

            logging.debug(f"[2F85] Set: 09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1")
            self.ser.write(
                b"\x09\x10\x03\xE8\x00\x03\x06\x01\x00\x00\x00\x00\x00\x72\xE1"
            )

            max_iteration = 10000

            for _ in range(max_iteration):
                logging.debug(f"[2F85] Set: 09 03 07 D0 00 01 85 CF")
                self.ser.write(b"\x09\x03\x07\xD0\x00\x01\x85\xCF")
                data_raw = self.ser.readline()
                logging.debug(f"[2F85] Status's Response: {data_raw}")
                if data_raw == b"\x09\x03\x02\x11\x00\x55\xD5":
                    logging.info(f"[2F85] Activate not complete")
                elif data_raw == b"\x09\x03\x02\x31\x00\x4C\x15":
                    logging.info(f"[2F85] Activate Complete")
                    return True
            logging.error(f"[2F85] Activate failed")
            return False
        except Exception as e:
            logging.error(f"[2F85] Activate failed: {e}")
            return False

    def inital_closed_pose(self, wait_time=0) -> bool:
        try:
            logging.info(f"[2F85] Initial gripper closed pose")
            position = b"\x50"  # 7F
            time.sleep(wait_time)  # calibrate:\x15
            speed = b"\xFF"  # 00:min;FF:max calibrate:\x15
            force = b"\xFF"  # 00:min;FF:max 15
            input = b"".join([b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00", position])
            input = b"".join([input, speed])
            input = b"".join([input, force])
            temp = self._mycrc(input)
            crc = (((temp << 8) | (temp >> 8)) & 0xFFFF).to_bytes(2, byteorder="big")
            write = b"".join([input, crc])

            self.ser.write(write)
            data_raw = self.ser.readline()  # bytes
            data = binascii.hexlify(data_raw)
            logging.info(f"[2F85] Initial end with response: {data.decode()}")
            return True
        except Exception as e:
            logging.error(f"[2F85] Initial failed: {e}")
            return False

    def on(self, wait_time=0) -> bool:
        logging.info(f"[2F85] Open gripper")
        try:
            # position = b'\x30' # open
            position = b"\x00"  # full open
            time.sleep(wait_time)  # calibrate:\x15
            speed = b"\xFF"  # 00:min;FF:max
            force = b"\xFF"  # 00:min;FF:max

            input = b"".join([b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00", position])
            input = b"".join([input, speed])
            input = b"".join([input, force])

            temp = self._mycrc(input)
            crc = (((temp << 8) | (temp >> 8)) & 0xFFFF).to_bytes(2, byteorder="big")
            write = b"".join([input, crc])

            self.ser.write(write)
            data_raw = self.ser.readline()  # bytes
            data = binascii.hexlify(data_raw)
            logging.info(f"[2F85] Open end with response: {data.decode()}")
            return True

        except Exception as e:
            logging.error(f"[2F85] Open failed: {e}")
            return False

    def close(self, wait_time=0) -> bool:
        logging.info(f"[2F85] Close gripper")
        try:
            position = b"\xFF"  # close
            time.sleep(wait_time)  # calibrate:\x15
            speed = b"\xFF"  # 00:min;FF:max calibrate:\x15
            force = b"\xFF"  # 00:min;FF:max 15
            input = b"".join([b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00", position])
            input = b"".join([input, speed])
            input = b"".join([input, force])
            temp = self._mycrc(input)
            crc = (((temp << 8) | (temp >> 8)) & 0xFFFF).to_bytes(2, byteorder="big")
            write = b"".join([input, crc])

            self.ser.write(write)
            data_raw = self.ser.readline()  # bytes
            data = binascii.hexlify(data_raw)
            logging.info(f"[2F85] Close end with response: {data.decode()}")
            return True

        except Exception as e:
            logging.error(f"[2F85] Close failed: {e}")
            return False

    def soft_close(self, wait_time=0) -> bool:
        logging.info(f"[2F85] Soft close gripper")
        try:
            position = b"\xF0"  # close
            time.sleep(wait_time)  # calibrate:\x15
            speed = b"\x15"  # 00:min;FF:max calibrate:\x15
            force = b"\x15"  # 00:min;FF:max 15
            input = b"".join([b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00", position])
            input = b"".join([input, speed])
            input = b"".join([input, force])
            temp = self._mycrc(input)
            crc = (((temp << 8) | (temp >> 8)) & 0xFFFF).to_bytes(2, byteorder="big")
            write = b"".join([input, crc])

            self.ser.write(write)
            data_raw = self.ser.readline()  # bytes
            data = binascii.hexlify(data_raw)
            logging.info(f"[2F85] Soft close end with response: {data.decode()}")
            return True

        except Exception as e:
            logging.error(f"[2F85] Soft close failed: {e}")
            return False

    def status(self) -> int: #可能可以用來控制夾爪寬度急走沒有成功夾取,測試中
        logging.info(f"[2F85] Get gripper status")
        self.ser.write(b"\x09\x03\x07\xD0\x00\x03\x04\x0E")
        data_raw = self.ser.readline()  # bytes
        data_show = binascii.hexlify(data_raw)
        print("Response:", data_show)
        gripper_status_mask = b"\xFF"
        gripper_status = bytes([data_raw[3] & gripper_status_mask[0]])

        gripper_status_bytes = int.from_bytes(gripper_status, "big")
        if (
            gripper_status_bytes & 0b00001000 == 0b00001000
            and gripper_status_bytes & 0b11000000 == 0b00000000
        ):
            logging.info(f"[2F85] No Object Detect (gripper moving)")
            return 0
        elif gripper_status_bytes & 0b11000000 == 0b01000000:
            logging.info(f"[2F85] Object Detect (opening)")
            return 1
        elif gripper_status_bytes & 0b11000000 == 0b10000000:
            logging.info(f"[2F85] Object Detect (closing)")
            return 2
        elif gripper_status_bytes & 0b11000000 == 0b11000000 or (
            gripper_status_bytes & 0b00001000 == 0b00000000
            and gripper_status_bytes & 0b11000000 == 0b00000000
        ):
            logging.info(f"[2F85] Object Detect (gripper stopped)")
            return 3
        else:
            logging.error(f"[2F85] Get gripper status failed")
            return -1

    def _mycrc(self, input):
        crc = 0xFFFF
        for byte in input:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc = crc >> 1
        return crc
