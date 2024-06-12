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
from typing import List

from robotic_drawing.control.robot_arm.yaskawa.fs100 import FS100
from robotic_drawing.control.robot_arm.yaskawa.yaskawa_config import *


class MH5L:
    def __init__(self, host, port):
        self.arm_speed = ARM_SPEED

        self.controller = FS100(host, port)

        self.init_x, self.init_y, self.init_z = INIT_Tx_BASE, INIT_Ty_BASE, INIT_Tz_BASE
        self.init_Rx, self.inite_Ry, self.init_RZ = (
            INIT_Rx_BASE,
            INIT_Ry_BASE,
            INIT_Rz_BASE,
        )

        # TODO: Known motivation
        self.angles = ANGLES
        self.angles = [int(angle * 10000) for angle in self.angles]

    def power_on(self):
        self.controller.power_on()

    def power_off(self):
        self.controller.power_off()

    def initialize_pose(self):
        response = self.controller.move_robot_pos(
            Tx=self.init_x,
            Ty=self.init_y,
            Tz=self.init_z,
            Rx=0,
            Ry=0,
            Rz=self.init_RZ,
            speed=self.arm_speed,
        )
        logging.info(
            f"[MH5L] Move to initial position: T=[{self.init_x}, {self.init_y}, {self.init_z}], R=[0, 0, {self.init_RZ}]"
        )
        if response != None:
            self.controller.wait_motion_end(mode="pose")
            logging.info(f"[MH5L] Move to initial end (success)")
        else:
            logging.error(f"[MH5L] Move to initial end (fail)")

    def go_initialize_pose(self):
        #! TODO: fix this
        logging.error(f"[MH5L] go_initialize_pose() is not implemented yet")

        # # go to lower position to avoid collide with ETH camera
        # self.controller.move_robot_pos(
        #     -8490, -365833, 200320, -1704993, -11611, -971988, self.arm_speed)
        # self.controller.wait_motion_end(mode="pose")
        # self.controller.move_robot_pos(
        #     self.init_x, self.init_y, self.init_z, -1704993, -11611, -971988, self.arm_speed)
        # self.controller.wait_motion_end(mode="pose")

    def move_to_pose(
        self, Tx, Ty, Tz, Rx=-180.0000, Ry=0, Rz=0, speed=ARM_SPEED
    ) -> bool:
        Tx, Ty, Tz = (int(Tx * 1000), int(Ty * 1000), int(Tz * 1000))
        Rx, Ry, Rz = (int(Rx * 10000), int(Ry * 10000), int(Rz * 10000))
        response = self.controller.move_robot_pos(Tx, Ty, Tz, Rx, Ry, Rz, speed=speed)
        logging.info(
            f"[MH5L] Move to position: T=[{Tx}, {Ty}, {Tz}], R=[{Rx}, {Ry}, {Rz}]"
        )
        if response != None:
            self.controller.wait_motion_end(mode="pose")
            logging.info(f"[MH5L] Move to position end (success)")
            return True
        else:
            logging.error(f"[MH5L] Move to position end (fail)")
            return False

    def move_to_joint_config(self, joint_config: List[int], speed=ARM_SPEED) -> bool:
        response = self.controller.move_robot_joint(
            joint_config[0],
            joint_config[1],
            joint_config[2],
            joint_config[3],
            joint_config[4],
            joint_config[5],
            speed,
        )
        logging.info(f"[MH5L] Move to joint config: {joint_config}")
        if response != None:
            self.controller.wait_motion_end(mode="joint")
            logging.info(f"[MH5L] Move to joint config end (success)")
            return True
        else:
            logging.error(f"[MH5L] Move to joint config end (fail)")
            return False

    def rotate_test(self):
        for i in range(0, len(self.angles), 4):
            self.controller.move_robot_pos(
                self.init_x,
                self.init_y,
                175715,
                -1704993,
                -11611,
                self.angles[i],
                self.arm_speed,
            )
            self.controller.wait_move_end()
            time.sleep(2)
