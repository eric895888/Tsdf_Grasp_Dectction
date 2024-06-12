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

import time

import cv2
import numpy as np

from .config import criteria, winSize, zeroZone


def corner_detection(
    image_paths, chessboard_intercorner_shape, chessboard_grid_size=1, show_result=True
):
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
    gray_shape = None
    result_images = []

    for fname in image_paths:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape
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
            # cv2.drawChessboardCorners(
            #     img,
            #     (chessboard_intercorner_shape[0], chessboard_intercorner_shape[1]),
            #     corners2,
            #     ret,
            # )
            # result_images.append(img)

    if show_result == True:
        cv2.destroyAllWindows()
        image_index = 0
        while True:
            if image_index < len(result_images):
                cv2.imshow("Image Viewer", result_images[image_index])
                image_index = image_index + 1
                time.sleep(1)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
    return objpoints, imgpoints, gray_shape


def undistortion(image, intrinsic_matrix, distortion_matrix):
    # h,  w = img.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coefficients, (w,h), 1, (w,h))
    # dst = cv2.undistort(img, intrinsic_matrix, distortion_coefficients, None, newcameramtx)
    dst = cv2.undistort(image, intrinsic_matrix, distortion_matrix, None, None)
    return dst
