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

from robotic_drawing.vision.camera.camera import Camera

class Realsense(Camera):
    
    def __init__(self, intrinsics, distortion_coefficients):
        super().__init__()
        self.intrinsics = intrinsics
        self.distortion_coefficients = distortion_coefficients
        self.streaming = False
        self._start_streaming()
        self.streaming = True
    
    def _enable_streaming(self):
        pass
    
    def _start_streaming(self):
        self._enable_streaming()
    
    def _stop_streaming(self):
        pass
    
    def _is_streaming(self):
        return self.streaming is True
    
    def get_color_frame(self):
        try:
            pass
        except Exception as e:
            pass
    
    def get_depth_frame(self):
        try:
            pass
        except Exception as e:
            pass
    
    def get_pixel_depth(self):
        pass
    
    def __delete__(self):
        self._stop_stream()