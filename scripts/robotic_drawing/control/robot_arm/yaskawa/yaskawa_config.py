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

HOST = ""  # e.g. 192.168.255.10
PORT = 11000

ARM_SPEED = 2000

# TODO: change this for TOOL 1
INIT_Tx_BASE = 10064  # -8490
INIT_Ty_BASE = -405458  # -365833
INIT_Tz_BASE = 203721  # 400320

# TODO: change this
INIT_Rx_BASE = 1762043  # -1704993  # TODO: check is this 0?
INIT_Ry_BASE = -11611  # TODO: check is this 0?
INIT_Rz_BASE = -971988

INIT_JOINT_CONFIGURE = [
    -133625,  # S
    -7888,  # L
    13852,  # U
    -2192,  # R
    -101744,  # B
    51436,  # T
]


STEP_DICT = {
    # TODO: Add missing key value pairs from c++ version code
    "GRIPPER_RESET": 0,
    "Get_GRIPPER_STATUS": 1,
    "GET_ROBOT_CART_POSITION": 2,
    "GET_ROBOT_PULES_POSITION": 15,
    "MOVE_POSITION": 3,
    "MOVE_JOINT": 4,
    "GRIPPER_OFF": 5,
    "GRIPPER_ON": 6,
    "CONVEYOR_A_OFF": 7,
    "CONVEYOR_A_ON": 8,
    "CONVEYOR_C_OFF": 11,
    "CONVEYOR_C_ON": 12,
    "CONVEYOR_D_OFF": 13,
    "CONVEYOR_D_ON": 14,
    "GET_ROBOT_CART_POSITION_TOOL4": 16,  # triangle P1
    "GET_ROBOT_CART_POSITION_TOOL5": 17,  # triangle P2
    "GET_ROBOT_CART_POSITION_TOOL6": 18,  # triangle P3
    "GET_ROBOT_CART_POSITION_TOOL7": 19,  # rectangle P1
    "GET_ROBOT_CART_POSITION_TOOL8": 20,  # rectangle P2
    "GET_ROBOT_CART_POSITION_TOOL9": 21,  # rectangle P3
    "GET_ROBOT_CART_POSITION_TOOL10": 22,  # circle P1
    "GET_ROBOT_CART_POSITION_TOOL11": 23,  # circle P2
    "GET_ROBOT_CART_POSITION_TOOL12": 24,  # circle P3
}

MOTION_CHECKING_PERIOD = 0.25  # unit: second

ANGLES = [
    -97,
    -119.5,
    -142,
    -164.5,
    -7,
    -29.5,
    -52,
    -74.5,
    -97,
    -119.5,
    -142,
    -164.5,
    -7,
    -29.5,
    -52,
    -74.5,
]

CALIBRATION_JOINT_CONFIG_LIST = [
    [-144961, -24786, 4151, -647, -97324, 55026],
    [-105237, -24786, -484, 20794, -97324, 55023],
    [-106676, -20961, -1109, 12425, -100524, 64782],
    [-177986, -11528, 5242, -6160, -99020, 62426],
    [-181697, -34098, -15667, -11328, -88503, 64324],
    [-170497, -54461, -23235, -9711, -84850, 50045],
    [-109068, -52549, -28610, 16788, -83689, 30559],
    [-133330, -50587, -23372, 2035, -82321, 37714],
    [-184772, -34543, -11763, -22577, -91810, 52523],
    [-195732, -45366, -18440, -27027, -91021, 56665],
    [-167066, -3332, 2896, -15161, -96084, 45184],
    [-159894, 21696, 17480, -3743, -107170, 65957],
    [-175786, 32284, 23905, -14637, -113966, 72481],
    [-167929, 38585, 33753, -4236, -112661, 85729],
    [-182556, -54526, -28351, -11695, -80727, 88688],
    [-152250, -27716, -5083, -1278, -94937, 77883],
    [-103041, -25491, -6991, 18637, -98020, 62300],
    [-107718, -15806, -4960, 17091, -96681, 56288],
    [-110489, -12821, -19513, 14127, -88192, 47281],
    [-110489, -2871, 25122, 17094, -113251, 52380],
    [-90956, 6599, 33229, 13698, -115628, 44132],
    [-182765, 40, 27223, -20532, -109889, 68006],
    [-198686, -26830, -6700, -23579, -97670, 77262],
    [-169931, 17260, 24804, -7871, -114432, 62848],
    [-177503, 26640, 34603, -5340, -116422, 66237],
    [-148041, -6458, 6009, 1861, -97796, 45716],
    [-199617, 30622, 22402, -32425, -115898, 57110],
    [-186572, 10917, 22536, -15289, -108378, 51635],
    [-148689, 21628, 33021, 667, -116438, 67942],
    [-154872, 18195, 24960, -4984, -112041, 66238],
]
