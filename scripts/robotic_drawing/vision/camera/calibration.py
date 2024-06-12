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

import cv2

from .util import corner_detection


def camera_calibration(image_paths, chessboard_intercorner_shape, chessboard_grid_size):
    assert type(image_paths) == list, "image_paths must be a list"
    assert len(image_paths) > 0, "image_paths must not be empty"

    objpoints, imgpoints, gray_shape = corner_detection(
        image_paths,
        chessboard_intercorner_shape,
        chessboard_grid_size,
        show_result=False,
    )
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape[::-1], None, None
    )
    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints
