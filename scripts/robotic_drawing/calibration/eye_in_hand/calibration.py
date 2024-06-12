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

import json
import logging

import cv2
from robotic_drawing.vision.camera.realsense.D400 import D400
from robotic_drawing.vision.camera.util import corner_detection

IMAGE_SAVE_FOLDER_PATH = "results"


def eye_in_hand_calibration(
    robot_arm,
    calibration_joint_config_list,
    chessboard_intercorner_shape,
    intrinsic_matrix,
    distortion_coefficients,
    method=cv2.CALIB_HAND_EYE_TSAI,
    speed=500,
):
    d400 = D400()
    d400.enable_stream("depth", 1280, 720, 30)
    d400.enable_stream("color", 1920, 1080, 30)
    d400.start(depth_unit=0.001)
    
    logging.info("Power on")
    robot_arm.power_on()

    calibration_record = {
        "pose_list": [],
        "image_paths": [],
        "extrinsic_matrix_list": [],
    }

    for index, joint_config in enumerate(calibration_joint_config_list):
        logging.info("Move to joint config: {}".format(joint_config))
        robot_arm.move_to_joint_config(joint_config, speed)
        calibration_record["pose_list"].append(robot_arm.get_cart_pose())

        chessboard_image = IMAGE()  # TODO cache image
        image_path = IMAGE_SAVE_FOLDER_PATH + f"/chessboard_{index}.jpg"
        chessboard_image.save(image_path)
        calibration_record["image_paths"].append(image_path)

    obj_points, img_points, gray_shape = corner_detection(
        calibration_record["image_paths"],
        chessboard_intercorner_shape,
        show_result=False,
    )
    # caculate extrinsic matrix
    ret, rvecs, tvecs = cv2.solvePnP(
        obj_points, img_points, intrinsic_matrix, distortion_coefficients
    )

    # eye in hand calibration
    R_gripper2base = calibration_record["pose_list"][:, 3:]  # / 1000.0
    t_gripper2base = calibration_record["pose_list"][:, 0:3]  # / 1000.0
    R_target2cam = rvecs
    t_target2cam = tvecs

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method
    )

    result = {
        "R_cam2gripper": R_cam2gripper.tolist(),
        "t_cam2gripper": t_cam2gripper.tolist(),
    }

    with open(f"{IMAGE_SAVE_FOLDER_PATH}/cam2gripper.json", "w") as f:
        json.dump(result, f)

    logging.info("Power off")
    robot_arm.power_off()

    return result