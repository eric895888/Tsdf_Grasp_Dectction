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

from robotic_drawing.control.tool.robotiq.gripper_2f85 import Gripper2F85


def test_gripper_control():
    logging.info("Init")
    gripper = Gripper2F85()
    logging.info("Connect")
    success = gripper.connect()
    logging.info("Reset")
    success = gripper.reset()

    try:
        while True:
            key_in = int(input("Input 0:finish 1:open 2:close 3:soft_close\n=>"))
            if key_in == 0:
                break
            elif key_in == 1:
                logging.info("Open")
                success = gripper.on()
            elif key_in == 2:
                logging.info("Close")
                success = gripper.close()
            elif key_in == 3:
                logging.info("Soft close")
                success = gripper.soft_close()
            else:
                logging.info("Undifine key in.")
    except:
        logging.info("Disconnect")
        success = gripper.disconnect()
    logging.info("Disconnect")
    success = gripper.disconnect()


if __name__ == "__main__":
    test_gripper_control()
