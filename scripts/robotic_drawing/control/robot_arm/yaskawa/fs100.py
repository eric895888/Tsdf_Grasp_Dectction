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

import logging
import time
from typing import List, Union

from robotic_drawing.control.robot_arm.yaskawa.yaskawa_config import *
from robotic_drawing.control.util.socket import TCPClient


class FS100:  # Controller of yaskawa robot
    def __init__(self, host, port):
        self.communication = None
        self.step_dic = STEP_DICT
        self.robot_info = None  # TODO: What is this?
        self.state = -1  # TODO: What is this?
        self.host = host
        self.port = port
        self.motion_checking_period = MOTION_CHECKING_PERIOD

    def power_on(self) -> bool:
        self.communication = TCPClient(self.host, self.port)
        success = self.communication.connect()
        if success == True:
            logging.info(f"[FS100] Power on success")
            return True
        else:
            logging.error(f"[FS100] Power on failed")
            return False

    def power_off(self) -> bool:
        success = self._send_close_command()
        if success == True:
            logging.info(f"[FS100] Power off success")
            return True
        else:
            logging.error(f"[FS100] Power off failed")
            return False

    def move_robot_pos(
        self, Tx: int, Ty: int, Tz: int, Rx: int, Ry: int, Rz: int, speed: int
    ) -> Union[List[str], None]:
        send_byte_data = self._create_command(
            Tx, Ty, Tz, Rx, Ry, Rz, speed, self.step_dic["MOVE_POSITION"]
        )
        success = self.communication.send(send_byte_data)
        if success == True:
            logging.info(f"[FS100] Send move position command success")
            response = self.communication.recv()
            # TODO: check format of "receive ' n ' : can't move"
            if response[0] == " n ":
                logging.error(
                    f"[FS100] Robot can't move to position: T = [{Tx},{Ty},{Tz}], R=[{Rx},{Ry},{Rz}]"
                )
                return None
            else:
                return response
        else:
            logging.error(f"[FS100] Send move position command failed")
            return None

    def move_robot_joint(
        self,
        joint1: int,
        joint2: int,
        joint3: int,
        joint4: int,
        joint5: int,
        joint6: int,
        speed: int,
    ) -> Union[List[str], None]:
        send_byte_data = self._create_command(
            joint1,
            joint2,
            joint3,
            joint4,
            joint5,
            joint6,
            speed,
            self.step_dic["MOVE_JOINT"],
        )
        # self.communication.send(send_byte_data)
        success = self.communication.send(send_byte_data)
        if success == True:
            logging.info(f"[FS100] Send move joint command success")
            response = self.communication.recv()
            # TODO: check format of "receive ' n ' : can't move"
            if response[0] == " n ":
                logging.error(
                    f"[FS100] Robot can't move to joint configure: [{joint1},{joint2},{joint3},{joint4},{joint5},{joint6}]"
                )
                return None
            else:
                return response
        else:
            logging.error(f"[FS100] Send move joint command failed")
            return None

    def get_robot_pose(self, tool_num=0):
        assert (
            0 == tool_num or 4 <= tool_num <= 12
        ), "tool_num should be in range [0, 12]"
        if tool_num == 0:
            send_byte_data = self._create_command(
                0, 0, 0, 0, 0, 0, 0, self.step_dic["GET_ROBOT_CART_POSITION"]
            )
        else:
            tool_num_dict = {
                4: self.step_dic["GET_ROBOT_CART_POSITION_TOOL4"],
                5: self.step_dic["GET_ROBOT_CART_POSITION_TOOL5"],
                6: self.step_dic["GET_ROBOT_CART_POSITION_TOOL6"],
                7: self.step_dic["GET_ROBOT_CART_POSITION_TOOL7"],
                8: self.step_dic["GET_ROBOT_CART_POSITION_TOOL8"],
                9: self.step_dic["GET_ROBOT_CART_POSITION_TOOL9"],
                10: self.step_dic["GET_ROBOT_CART_POSITION_TOOL10"],
                11: self.step_dic["GET_ROBOT_CART_POSITION_TOOL11"],
                12: self.step_dic["GET_ROBOT_CART_POSITION_TOOL12"],
                13: self.step_dic["GET_ROBOT_CART_POSITION_TOOL13"],
                14: self.step_dic["GET_ROBOT_CART_POSITION_TOOL14"],
            }
            send_byte_data = self._create_command(
                0, 0, 0, 0, 0, 0, 0, tool_num_dict[tool_num]
            )

        success = self.communication.send(send_byte_data)
        if success == True:
            logging.info(f"[FS100] Send get robot position command success")
            response = self.communication.recv()
            logging.info(f"[FS100] Get robot position: {response}")
            if response is None:
                logging.error(f"[FS100] Receive robot position failed")
                return None
            else:
                # TODO: Unknow motivation
                for i in response:
                    if str(i).find("s") != -1 or str(i).find("n") != -1:
                        return response

                # TODO : Unknow motivation
                # self.robot_info = [int(i) / 1000 for i in response]
                # Tx, Ty, Tz = self.robot_info[0], self.robot_info[1], self.robot_info[2]
                # Rx, Ry, Rz = self.robot_info[3], self.robot_info[4], self.robot_info[5]
                response = [int(i) for i in response]
                response[0], response[1], response[2] =  (response[0] / 1000, response[1] / 1000, response[2] / 1000)
                response[3], response[4], response[5] =  (response[3] / 10000, response[4] / 10000, response[5] / 10000)
                return response

    def get_robot_pulse(self):
        send_byte_data = self._create_command(
            0, 0, 0, 0, 0, 0, 0, self.step_dic["GET_ROBOT_PULES_POSITION"]
        )
        success = self.communication.send(send_byte_data)
        if success == True:
            logging.info(f"[FS100] Send get robot pulse command success")
            response = self.communication.recv()
            logging.info(f"[FS100] Get robot pulse: {response}")
            if response is None:
                logging.error(f"[FS100] Receive robot pulse failed")
                return None
            else:
                for i in response:
                    if str(i).find("s") != -1 or str(i).find("n") != -1:
                        return response

                # self.robot_info = [int(i) for i in response]
                # joint1, joint2, joint3, joint4, joint5, joint6 = self.robot_info
                return response
        else:
            logging.error(f"[FS100] Send get robot pulse command failed")
            return None

    def wait_motion_end(self, mode: str) -> None:
        assert mode == "joint" or mode == "pose", "mode should be 'pulse' or 'pose'"
        if mode == "pose":
            pos1 = self.get_robot_pose()
            #! TODO: Effect speed
            time.sleep(self.motion_checking_period)
            pos2 = self.get_robot_pose()
            while pos1 != pos2:
                pos1 = pos2
                time.sleep(self.motion_checking_period)
                pos2 = self.get_robot_pose()
        elif mode == "joint":
            pulse1 = self.get_robot_pulse()
            time.sleep(self.motion_checking_period)
            pulse2 = self.get_robot_pulse()
            while pulse1 != pulse2:
                pulse1 = pulse2
                time.sleep(self.motion_checking_period)
                pulse2 = self.get_robot_pulse()
        logging.info(f"[FS100] Motion end (by waiting)")

    def _send_close_command(self) -> bool:
        # if self.communication.is_connected():
        send_data = "c"
        success = self.communication.send(send_data)
        if success == True:
            logging.info(f"[FS100] Send close command success")
            return True
        else:
            logging.error(f"[FS100] Send close command failed")
            return False

    def _create_command(
        self,
        d1: int,
        d2: int,
        d3: int,
        d4: int,
        d5: int,
        d6: int,
        speed: int,
        step: int,
    ) -> str:
        return f"{d1},{d2},{d3},{d4},{d5},{d6},{speed},{step};"

    # def check_motion_end(self, mode: str, input) -> bool:
    #     assert mode == "joint" or mode == "pose", "mode should be 'pulse' or 'pose'"

    #     if mode == "joint":
    #         self.get_robot_pulse()
    #     elif mode == "pose":
    #         self.get_robot_pose()
    #     logging.info(f"[FS100] Motion end (by checking)")
    #     return self.robot_info == input

    # def gripper_on(self)->List[str]:
    #     send_byte_data = self._create_command_bytes(0, 0, 0, 0, 0, 0, 0, self.step_dic['GRIPPER_ON'])
    #     success = self.communication.send(send_byte_data)
    #     if success == True:
    #         logging.info(f"[FS100] Send gripper on command success")
    #         response = self.communication.recv()
    #         return response
    #     else:
    #         logging.error(f"[FS100] Send gripper on command failed")
    #         return None

    # def gripper_off(self)->List[str]:
    #     send_byte_data = self._create_command_bytes(0, 0, 0, 0, 0, 0, 0, self.step_dic['GRIPPER_OFF'])
    #     success = self.communication.send(send_byte_data)
    #     if success == True:
    #         logging.info(f"[FS100] Send gripper off command success")
    #         response = self.communication.recv()
    #         return response
    #     else:
    #         logging.error(f"[FS100] Send gripper off command failed")
    #         return None

    # def gripper_reset(self):
    #     send_byte_data = self._create_command_bytes(0, 0, 0, 0, 0, 0, 0, self.step_dic['GRIPPER_RESET'])
    #     success = self.communication.send(send_byte_data)
    #     if success == True:
    #         logging.info(f"[FS100] Send gripper reset command success")
    #         response = self.communication.recv()
    #         return response
    #     else:
    #         logging.error(f"[FS100] Send gripper reset command failed")
    #         return None

    # def get_gripper_status(self):
    #     send_byte_data = self._create_command_bytes(0, 0, 0, 0, 0, 0, 0, self.step_dic['Get_GRIPPER_STATUS'])
    #     success = self.communication.send(send_byte_data)
    #     if success == True:
    #         logging.info(f"[FS100] Send get gripper status command success")
    #         response = self.communication.recv()
    #     else:
    #         logging.error(f"[FS100] Send get gripper status command failed")
    #         return None

    #     for i in response:
    #         if str(i).find('s') != -1 or str(i).find('n') != -1:
    #             return response

    #     # TODO: Unknow motivation
    #     self.robot_info = [int(i) for i in response]  # busy 1 0 0 0 0 0
    #     logging.info(f"[FS100] Get gripper status: {self.robot_info}")

    #     return response
