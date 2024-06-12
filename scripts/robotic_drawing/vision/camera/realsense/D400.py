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

# References:
# [1] https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-tutorial-1-depth.py

import logging

import cv2
import numpy as np
import pyrealsense2 as rs
from robotic_drawing.vision.camera.realsense.realsense import Realsense


class D400(Realsense):
    def __init__(self):
        logging.info("[D400] Initializing")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_start = False
        self.depth_sensor = None
        logging.info("[D400] Initialized")

    def enable_stream(self, stream_type, width=640, height=480, fps=30):
        assert stream_type in ["depth", "color"]  # , 'infrared']
        stream_type_dict = {
            "depth": rs.stream.depth,
            "color": rs.stream.color,
            # "infrared": rs.stream.infrared,
        }
        stream_data_type_dict = {
            "depth": rs.format.z16,
            "color": rs.format.bgr8,
        }
        logging.info(
            "[D400] Enabling stream: "
            + str(stream_type)
            + " "
            + str(width)
            + "x"
            + str(height)
            + "@"
            + str(fps)
            + "fps"
        )
        self.config.enable_stream(
            stream_type_dict[stream_type],
            width,
            height,
            stream_data_type_dict[stream_type],
            fps,
        )

    def start(self, depth_unit=0.001):
        # Start streaming
        profile = self.pipeline.start(self.config)
        self.pipeline_start = True
        self.depth_sensor = profile.get_device().first_depth_sensor()
        if self.depth_sensor.supports(rs.option.depth_units):
            self.depth_sensor.set_option(rs.option.depth_units, depth_unit)
            logging.info(f"[D400] Set depth unit to {depth_unit}")

        try:
            logging.info("[D400] Start streaming")
            # while True:
            #     # Wait for a coherent pair of frames: depth and color
            #     frames = self.pipeline.wait_for_frames()
            #     depth_frame = frames.get_depth_frame()
            #     color_frame = frames.get_color_frame()
            #     if not depth_frame or not color_frame:
            #         continue

            #     # Convert images to numpy arrays
            #     depth_image = np.asanyarray(depth_frame.get_data())
            #     color_image = np.asanyarray(color_frame.get_data())

            #     # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            #     depth_colormap = cv2.applyColorMap(
            #         cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            #     )

            #     depth_colormap_dim = depth_colormap.shape
            #     color_colormap_dim = color_image.shape

            #     # If depth and color resolutions are different, resize color image to match depth image for display
            #     if depth_colormap_dim != color_colormap_dim:
            #         resized_color_image = cv2.resize(
            #             color_image,
            #             dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
            #             interpolation=cv2.INTER_AREA,
            #         )
            #         images = np.hstack((resized_color_image, depth_colormap))
            #     else:
            #         images = np.hstack((color_image, depth_colormap))

            #     # Show images
            #     cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            #     cv2.imshow("RealSense", images)
            #     cv2.waitKey(1)

        except Exception as e:
            logging.error(f"[D400] Streaming error: {e}")
            if self.pipeline_start == True:
                self.pipeline.stop()
                self.pipeline_start = False

    def stop(self):
        self.pipeline.stop()
        logging.info("[D400] Stop streaming")

    def get_frame(self):
        logging.info("[D400] Get frame")
        try:
            while True:
                self.frames = self.pipeline.wait_for_frames()
                self.align = rs.align(rs.stream.color)
                self.frames = self.align.process(self.frames)
                self.depth_frame = self.frames.get_depth_frame()
                self.color_frame = self.frames.get_color_frame()
                if not self.depth_frame or not self.color_frame:
                    continue
                else:
                    # rs2.video_frame need get_data() to change to img
                    #     # Convert images to numpy arrays
                    depth_scale = float(self.depth_sensor.get_depth_scale())
                    print(f"ds {depth_scale}")
                    depth_image = np.asanyarray(self.depth_frame.get_data())
                    depth_image = depth_image.astype(float) * depth_scale
                    color_image = np.asanyarray(self.color_frame.get_data())
                    # return self.color_frame, self.depth_frame
                    return color_image, depth_image

        except Exception as e:
            logging.error(f"[D400] Get frame error: {e}")
            if self.pipeline_start == True:
                self.pipeline.stop()
                self.pipeline_start = False

    def __del__(self):
        if self.pipeline_start == True:
            self.pipeline.stop()
            self.pipeline_start = False
