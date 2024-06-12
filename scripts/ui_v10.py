# Copyright 2024 tc-haung
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

import os
import sys
import cv2
import json
import time
import logging
import numpy as np
from pathlib import Path
import pyrealsense2 as rs
from datetime import datetime
from threading import Thread, Event
from collections import namedtuple, deque
from scipy.spatial.transform import Rotation

from robotic_drawing.control.robot_arm.yaskawa.mh5l import MH5L

HOST = "192.168.255.10"
PORT = 11000

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QSizePolicy

# logging.basicConfig(level=logging.INFO)

CALIBRATION_PARAMS_SAVE_DIR = (
    Path(__file__).parent
    / f"calibration_params/{datetime.now().strftime('%Y_%m_%d')}"
)

CHESSBOARD_GRID_SIZE = 20  # mm
CHESSBOARD_INTERCORNER_SHAPE = (11, 8)#(10, 7)

# Joint T + 52680 after install bin on flang
INIT_JOINT_CONFIGURE = [
    -126330,
    -18970,
    14748,
    -1553,
    -101441,
    46069 + 52680
]

CALIBRATION_JOINT_CONFIG_LIST = [
        [-126330, -17380, 18559, -1574, -103231, 46035 + 52680],
        [-126330, -15620, 22679, -1599, -105130, 46000 + 52680],
        [-126330, -19968, 12272, -1538, -100260, 46100 + 52680],
        [-126331, -21176, 9196, -1523, -98770, 46128 + 52680],
        [-126331, -22625, 5351, -1506, -96872, 46162 + 52680],

        [-125771, -27217, 2700, -1494, -97709, 45993 + 52680],
        [-125308, -30542, 918, -1486, -98413, 45746 + 52680],
        [-126643, -19741, 7128, -1506, -96425, 46248 + 52680],

        [-124435, -65693, -26097, -225, -80889, 98928],
        [-124435, -83617, -28152, -225, -80890, 98927],
        [-124515, -83620, -36222, -225, -80891, 98925],
        [-124515, -83625, -36221, -225, -75732, 98940],
        [-124307, -58157, -36221, -225, -75723, 98940],

        [-125475, 21059, 27919, 639, -111616, 98947],
        [-125475, 25475, 24709, 639, -111839, 98952],
        [-125476, 30451, 19685, 639, -110237, 98951],
        [-125476, 19091, 26735, 639, -110234, 100150],
        [-125476, 11027, 26734, 639, -111177, 110148],

        [-129859, -22717, 5295, -1607, -96949, 47483 + 52680],
        [-134499, -22815, 5237, -1742, -97049, 49223 + 52680],
        [-138126, -22746, 5278, -1844, -97108, 50580 + 52680],
        [-118899, -21643, 5946, -1270, -96594, 43359 + 52680],
        [-115462, -21049, 6310, -1158, -96455, 42076 + 52680],
        [-110734, -20016, 6953, -1001, -96242, 40317 + 52680],

        # [-121423, -21872, 5814, 1920, -97233, 46792 + 52680],
        # [-116624, -21254, 5941, 5466, -97667, 45692 + 52680],
        # [-112504, -20622, 5899, 8535, -98128, 44763 + 52680],
        # [-106456, -19506, 5558, 13106, -98953, 43407 + 52680],
        # [-100990, -18276, 4922, 17335, -99856, 42187 + 52680],

        # [-131410, -22778, 4983, -5396, -96668, 49114 + 52680],
        # [-138023, -23603, 3708, -10232, -96233, 50699 + 52680],
        # [-143723, -23715, 2600, -14439, -96305, 52071 + 52680],
        # [-148419, -23678, 1474, -17941, -96483, 53222 + 52680],
        # [-152934, -23521, 189, -21356, -96759, 54350 + 52680],

        # [-123273, -27616, 1904, -19, -93723, 44490 + 52680],
        # [-124567, -18749, 7526, -978, -99530, 44760 + 52680],
        # [-125241, -13283, 10797, -1518, -103104, 44858 + 52680],
        
        # [-123320, -23885, 4317, 14, -96182, 40174 + 52680],
        # [-124838, -23594, 4487, -1276, -96376, 53051 + 52680],
        INIT_JOINT_CONFIGURE
    ]
SPEED = 300

def save_json(data: dict, save_file_path: str):
    with open(save_file_path, "w") as f:
        json.dump(data, f, indent=4)

class D400Thread(QThread):
    current_color_frame = Signal(QImage)
    current_depth_frame = Signal(QImage)
    catch_color_image = Signal(QImage)
    catch_depth_image = Signal(QImage)
    
    def __init__(self):
        super().__init__()

        self.pipeline = rs.pipeline()
        self.config = self._set_config()
        self.align = rs.align(rs.stream.color)
        self._streaming_event = Event()
        self.depth_scale = None
        # self.color_image = None
        # self.depth_image = None
    
    def run(self):
        profile = self.pipeline.start(self.config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self._streaming_event.set()
        logging.info("[D400] Pipeline started")
        try:
            while self._streaming_event.is_set():
                frames = self.pipeline.wait_for_frames()
                frames = self.align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not depth_frame or not color_frame:
                    continue

                color_image = self._frame2image(color_frame)
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                self.color_image = color_image.copy()

                self.depth_image = self._frame2image(depth_frame) 
                
                color_image = self._draw_center_cross(color_image)

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
                
                self.current_color_frame.emit(self._array_image2qimage(color_image, 640, 480))
                self.current_depth_frame.emit(self._array_image2qimage(depth_colormap, 640, 480))
        
        except Exception as e:
            logging.error(f"[D400] ERROR: {e}")

        finally:
            if self._streaming_event.is_set():
                self._streaming_event.clear()
                self.pipeline.stop()
                logging.info("[D400] Pipeline stopped")
            sys.exit(-1)
    
    def _set_config(self):
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        return config
    
    def _frame2image(self, frame):
        image = np.asanyarray(frame.get_data())
        return image
    
    def _array_image2qimage(self, array, new_size_w=None, new_size_h=None):
        h, w, ch = array.shape
        bytes_per_line = ch * w
        qimage = QImage(array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        if new_size_w and new_size_h:
            qimage = qimage.scaled(new_size_w, new_size_h, Qt.KeepAspectRatio)
        return qimage
    
    def _draw_center_cross(self, color_image):
        color_image_copy = color_image.copy()
        h, w = color_image_copy.shape[:2]
        center_h, center_w = h // 2, w // 2
        cv2.line(color_image_copy, (center_w, 0), (center_w, h), (0, 0, 255), 1)
        cv2.line(color_image_copy, (0, center_h), (w, center_h), (0, 0, 255), 1)
        return color_image_copy
    
    def catch_color_depth_image(self, save_dir=None, index=None):

        color_image, depth_image = self.color_image, self.depth_image
        depth_image = depth_image.astype(float) * float(self.depth_scale) * 1000
        self.catch_color_image.emit(self._array_image2qimage(color_image, 640, 480))
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        if save_dir is not None and index is not None:
            cv2.imwrite(f"{save_dir}/color_{index}.jpg", color_image)
            # cv2.imwrite(f"{save_dir}/depth_{index}.jpg", depth_image)
        return color_image, depth_image

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Title and dimensions
        self.setWindowTitle("Camera and Eye-in-hand Calibration")
        self.setGeometry(0, 0, 1280, 900)
        # Create a label for the display camera
        self.label_color_image = QLabel(self)
        self.label_color_image.setFixedSize(640, 480)
        self.label_depth_image = QLabel(self)
        self.label_depth_image.setFixedSize(640, 480)
        self.label_result_image = QLabel(self)
        self.label_result_image.setFixedSize(640, 480)
        
        self.camera_thread = D400Thread()
        self.camera_thread.finished.connect(self.close)
        self.camera_thread.current_color_frame.connect(self.setColorImage)
        self.camera_thread.current_depth_frame.connect(self.setDepthImage)
        self.camera_thread.catch_color_image.connect(self.setResultImage)
         
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Color Image"))
        left_layout.addWidget(self.label_color_image)
        left_layout.addWidget(QLabel("Depth Image"))
        left_layout.addWidget(self.label_depth_image)
        
        buttons_layout = QHBoxLayout()
        self.button1 = QPushButton("Start Calibration")
        self.button1.clicked.connect(self._start_calibration)
        self.button2 = QPushButton("Test Calibration")
        self.button2.clicked.connect(self._test_calibration)
        INIT_JOINT_CONFIGURE
        self.button1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        buttons_layout.addWidget(self.button1)
        self.button2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        buttons_layout.addWidget(self.button2)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Result"))
        right_layout.addWidget(self.label_result_image)
        right_layout.addLayout(buttons_layout)
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
                # robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
        # logging.info("[Robot] Move to initial pose by joint configure")
        # time.sleep(1)
        # tcp_pose = robot_arm.controller.get_robot_pose(tool_num=0)
        # print(f"BACK TO INIT PoSe[{index}]: {tcp_pose}!!!")
        self._start()

    def closeEvent(self, event):
        self._kill_thread()
        event.accept()
    
    def _start(self):
        logging.info("[Main] Start streaming")
        self.camera_thread.start()

    @Slot(QImage)
    def setColorImage(self, image):
        self.label_color_image.setPixmap(QPixmap.fromImage(image))
    
    @Slot(QImage)
    def setDepthImage(self, image):
        self.label_depth_image.setPixmap(QPixmap.fromImage(image))
    
    @Slot(QImage)
    def setResultImage(self, image):
        self.label_result_image.setPixmap(QPixmap.fromImage(image))
    
    @Slot()
    def _kill_thread(self):
        self.camera_thread._streaming_event.clear()
        self.camera_thread.terminate()
        logging.info("[Main] Thread finishing...")
        # self.th.terminate()
        # self.th.wait()
    
    @Slot()
    def _start_calibration(self):
        logging.info("[Button] Start Calibration")
        self.calibration_thread = Thread(target=calibration, args=(INIT_JOINT_CONFIGURE, CALIBRATION_JOINT_CONFIG_LIST, self.camera_thread))
        self.calibration_thread.daemon = True
        self.calibration_thread.start()
    
    @Slot()
    def _test_calibration(self):
        logging.info("[Button] Test Calibration")
        self.test_calibration_thread = Thread(target=test_calibration, args=(self.camera_thread,))
        self.test_calibration_thread.daemon = True
        self.test_calibration_thread.start()
    # @Slot()
    # def _catch_image(self):
    #     logging.info("[Button] Catch Image")
    #     self.th.catch_color_depth_image()

def corner_detection(
    color_image_list,
    chessboard_intercorner_shape,
    chessboard_grid_size=1,
    show_result=True,
):
    # termination criteria
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros(
        (chessboard_intercorner_shape[0] * chessboard_intercorner_shape[1], 3),
        np.float32,
    )
    objp[:, :2] = (
        np.mgrid[
            0 : chessboard_intercorner_shape[0], 0 : chessboard_intercorner_shape[1]
        ].T.reshape(-1, 2)
        * chessboard_grid_size
    )
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for i, color_image in enumerate(color_image_list):
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            (chessboard_intercorner_shape[0], chessboard_intercorner_shape[1]),
            None,
        )
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria)
            imgpoints.append(corners2)
            if show_result is True:
                cv2.drawChessboardCorners(
                    color_image,
                    (chessboard_intercorner_shape[0], chessboard_intercorner_shape[1]),
                    corners2,
                    ret,
                )
                # cv2.imshow(f"Corner [{i}]", color_image)
                # cv2.waitKey(0)
        else:
            logging.error("No corners found")
            exit(1)

    return objpoints, imgpoints


def calibration(init_joint_configure, calibration_joint_config_list, camera_thread):

    logging.info("[Calibration] Start")
    
    
    # create folder if not exist
    CALIBRATION_PARAMS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    robot_arm = MH5L(host=HOST, port=PORT)
    logging.info("[Robot] Init")
    robot_arm.power_on()
    logging.info("[Robot] Power on")


    robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
    logging.info("[Robot] Move to initial pose by joint configure")
    tcp_pose = robot_arm.controller.get_robot_pose(tool_num=0)
    
    color_image_list = []
    R_gripper2base_list = []
    t_gripper2base_list = []
    
    
    for index, joint_config in enumerate(calibration_joint_config_list):
        robot_arm.move_to_joint_config(joint_config, SPEED)
        logging.info("[Robot] Move to joint config: {}".format(joint_config))
        tcp_pose = robot_arm.controller.get_robot_pose(tool_num=0)
        print(f"CALIBRATION POSE[{index}]: {tcp_pose}")
        tvec = tcp_pose[:3]
        x_angle, y_angle, z_angle = tcp_pose[3:]
        rvec = Rotation.from_euler(
            "xyz", [x_angle, y_angle, z_angle], degrees=True
        ).as_rotvec(degrees=False)
        R_gripper2base_list.append(rvec)
        t_gripper2base_list.append(tvec)
        time.sleep(1)

        color_image, depth_image = camera_thread.catch_color_depth_image(
            str(CALIBRATION_PARAMS_SAVE_DIR),
            index,
        )
        
        color_image_list.append(color_image)
        logging.info("[Camera] Catch color and depth image")

        # robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
        # logging.info("[Robot] Move to initial pose by joint configure")
        # time.sleep(1)
        # tcp_pose = robot_arm.controller.get_robot_pose(tool_num=0)
        # print(f"BACK TO INIT PoSe[{index}]: {tcp_pose}!!!")
    

    robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
    logging.info("[Robot] Move to initial pose by joint configure")

    robot_arm.power_off()
    logging.info("[Robot] Power off")

    objpoints, imgpoints = corner_detection(
        color_image_list,
        chessboard_intercorner_shape=CHESSBOARD_INTERCORNER_SHAPE,
        chessboard_grid_size=CHESSBOARD_GRID_SIZE,  # mm
        show_result=True,
    )
    
    image_size = (1280, 720)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    img = color_image_list[-1].copy()
    img = cv2.drawFrameAxes(img, mtx, dist, rvecs[-1], tvecs[-1], length=10)
    cv2.imwrite('drawFrameAxes.jpg', img)
    # R_gripper2base_.size() == t_gripper2base_.size() && R_target2cam_.size() == t_target2cam_.size() && R_gripper2base_.size() == R_target2cam_.size()
    
    R_target2cam_list, t_target2cam_list = np.array(rvecs), np.array(tvecs)
    R_gripper2base_list, t_gripper2base_list = np.array(R_gripper2base_list), np.array(t_gripper2base_list)

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_list,
        t_gripper2base_list,
        R_target2cam_list,
        t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # print(f"R_cam2gripper: {R_cam2gripper.shape}")
    # print(f"t_cam2gripper: {t_cam2gripper.shape}")
    # print(f"t_cam2gripper: {t_cam2gripper}")
    # print(f"R_target2cam: {R_target2cam_list.shapeINIT_JOINT_CONFIGURE}")
    # print(f"t_target2cam: {t_target2cam_list.shape}")(
    # print(f"t_target2cam: {t_target2cam_list}")
    # print(f"R_gripper2base: {R_gripper2base_list.shape}")
    # print(f"t_gripper2base: {t_gripper2base_list.shape}")
    # print(f"t_gripper2base: {t_gripper2base_list}")

    with open('calibration.npy', 'wb') as f:
        np.save(f, R_cam2gripper)
        np.save(f, t_cam2gripper)
        np.save(f, R_target2cam_list)
        np.save(f, t_target2cam_list)
        np.save(f, R_gripper2base_list)
        np.save(f, t_gripper2base_list)
    
    # R_cam2gripper: (3, 3)
    # t_cam2gripper: (3, 1)
    # R_target2cam: (7, 3, 1)
    # t_target2cam: (7, 3, 1)
    # R_gripper2base: (7, 3)
    # t_gripper2base: (7, 3)

    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.T
    # print("t_cam2gripper", t_cam2gripper, T_cam2gripper[:3, 3])

    T_target2cam = np.eye(4)
    T_target2cam[:3, :3] = Rotation.from_rotvec(R_target2cam_list[-1].T).as_matrix()
    T_target2cam[:3, 3] = t_target2cam_list[-1].T
    # print("t_target2cam_list[-1]", t_target2cam_list[-1], T_target2cam[:3, 3])

    T_gripper2base = np.eye(4)
    T_gripper2base[:3, :3] = Rotation.from_rotvec(R_gripper2base_list[-1].T).as_matrix()
    T_gripper2base[:3, 3] = t_gripper2base_list[-1]
    # print("t_gripper2base_list[-1]", t_gripper2base_list[-1], T_gripper2base[:3, 3])

    result = {
        "T_cam2gripper": T_cam2gripper.tolist(),
        "T_target2cam": T_target2cam.tolist(),
        "T_gripper2base": T_gripper2base.tolist(),
        "intrinsic_matrix": mtx.tolist(),
        "distortion_coefficients" :dist.tolist()
    }
    # print(f"result: {result}")
    save_json(data=result, save_file_path=str(CALIBRATION_PARAMS_SAVE_DIR / 'calibration_params.json'))

    logging.info("[Calibration] End")
    # print("[Calibration] End")
    return

def test_calibration(camera_thread):

    with open(str(CALIBRATION_PARAMS_SAVE_DIR / 'calibration_params.json')) as f:
        calibration_params = json.load(f)
    
    T_cam2gripper = np.array(calibration_params["T_cam2gripper"])
    T_target2cam = np.array(calibration_params["T_target2cam"])
    T_gripper2base = np.array(calibration_params["T_gripper2base"])
    intrinsic_matrix = np.array(calibration_params["intrinsic_matrix"])
    distortion_coefficients = np.array(calibration_params["distortion_coefficients"])

     
    robot_arm = MH5L(host=HOST, port=PORT)
    logging.info("[Robot] Init")
    robot_arm.power_on()
    logging.info("[Robot] Power on")

    robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, 50)
    logging.info("[Robot] Move to initial pose by joint configure")

    """
    for i in range(3):
        break
        robot_arm.move_to_joint_config([
    -126330,
    -18970,
    14748,
    -1553,
    -101441,
    46069
    ], SPEED)
        time.sleep(2)
        logging.info("[Robot] Move to initial pose by joint configure")
        robot_arm.move_to_joint_config([
    -126330,
    -18970,
    14748,
    -1553,
    -101441,
    46069 + 50000
], SPEED)
        time.sleep(0.5)
        robot_arm.move_to_joint_config([
    -126330,
    -18970,
    14748,
    -1553,
    -101441,
    46069 - 50000
], SPEED)
        time.sleep(0.5)
    robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
    logging.info("[Robot] Move to initial pose by joint configure")
    """

    color_image, depth_image = camera_thread.catch_color_depth_image()
    
    color_image_undistorted = cv2.undistort(color_image, intrinsic_matrix, distortion_coefficients)
    # color_image_undistorted = color_image
    objpoints, imgpoints = corner_detection(
        [color_image_undistorted],
        chessboard_intercorner_shape=CHESSBOARD_INTERCORNER_SHAPE,
        chessboard_grid_size=CHESSBOARD_GRID_SIZE,  # mm
        show_result=True,
    )
    corner_index = 0#49
    target_point_image = imgpoints[0][corner_index][0]
    print(f"target_point_image: {target_point_image}")
    print(f"depth image: {depth_image.shape}")
    target_depth = depth_image[int(target_point_image[1])][int(target_point_image[0])]
    print(f"target_depth: {target_depth}")
    # return
    # depth = 445 # mm
    offset = 0
    depth = target_depth

    save_image = color_image_undistorted.copy()
    save_image = cv2.circle(save_image, center=(int(target_point_image[0]), int(target_point_image[1])), radius=10, color=(255, 0, 0))
    cv2.imwrite('draw_point.jpg', save_image)

    fx, fy, cx, cy = intrinsic_matrix[0][0], intrinsic_matrix[1][1], intrinsic_matrix[0][2], intrinsic_matrix[1][2]
    target_point_x = (target_point_image[0] - cx) / fx * depth
    target_point_y = (target_point_image[1] - cy) / fy * depth
    target_point_z = depth
    target_point_camera = np.array([target_point_x, target_point_y, target_point_z, 1]).T
    # print(f"target_point_camera: {target_point_camera}")
    # print(f"T_cam2gripper: {T_cam2gripper}")
    
    
    target_point_gripper = T_cam2gripper @ target_point_camera
    # print(f"target_point_gripper: {target_point_gripper}")
    # print(f"T_gripper2base: {T_gripper2base}")

    target_point_base = T_gripper2base @ target_point_gripper
    print(f"target_point_base: {target_point_base}")
    

    
    robot_arm.move_to_pose(Tx=target_point_base[0], Ty=target_point_base[1], Tz = target_point_base[2] + offset, speed=200)
    tcp_pose = robot_arm.controller.get_robot_pose(tool_num=0)
    tcp_pose = [i for i in tcp_pose]
    print(f"tcp: {tcp_pose}=======================================================")
    
    robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
    logging.info("[Robot] Move to initial pose by joint configure")

    robot_arm.power_off()
    logging.info("[Robot] Power off")
  

if __name__ == "__main__":
    app = QApplication()
    try:
        logging.info("[Main] Start streaming")
        w = MainWindow()
        w.show()
    except KeyboardInterrupt:
        logging.warning("[Main] KeyboardInterrupt")      
    except Exception as e:
        logging.error(f"[Main] ERROR: {e}")
    finally:
        logging.info("[Main] End of program")
        sys.exit(app.exec())