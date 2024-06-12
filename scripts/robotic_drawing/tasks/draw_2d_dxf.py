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

import logging
import numpy as np
from scipy.optimize import minimize
from spatialmath import SE3
from robotic_drawing.tasks.task import Task


def robot_arm_setup():
    pass

def robot_arm_shotdown():
    pass

def camera_setup():
    pass

def camera_shotdown():
    pass

def detect_paper():
    pass

def parse_dxf(dxf_file_path):
    pass

def image2camera(u:int, v:int, depth:float, intrinsic_matrix)->(float, float):
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    z = depth
    return x, y, z

def camera2world(x:float, y:float, z:float, camera2world_matrix)->SE3:
    camera2world_matrix = SE3(camera2world_matrix)
    camera2world_matrix_inv = camera2world_matrix.inv()
    camera_coordinate = SE3([x, y, z])
    world_coordinate = camera2world_matrix_inv * camera_coordinate
    return world_coordinate

def caculate_plan_equation_coefficients(origin, points:list):
    def distance_to_plane(point, plane_coeffs):
        a, b, c, d = plane_coeffs
        x, y, z = point
        return (a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)

    def total_distance_squared(points, plane_coeffs):
        return sum(distance_to_plane(p, plane_coeffs)**2 for p in points)
    
    def plane_constraint(plane_coeffs, origin_point):
        a, b, c, d = plane_coeffs
        x, y, z = origin_point
        return a * x + b * y + c * z + d
    
    initial_guess = [1, 1, 1, 1]
    constraints = {'type': 'eq', 'fun': plane_constraint, 'args': (origin,)}

    result = minimize(total_distance_squared, initial_guess, args=(points,), constraints=constraints, method='SLSQP')

    assert result.success, "The solver failed to converge!"
    fitted_plane = result.x
    a, b, c, d = fitted_plane
    return a, b, c, d

def R_between_coordinate_system(A, B):
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)

    R = np.dot(B.T, np.linalg.inv(A.T))
    return R


def R_metrix(u_x, u_y, u_z):
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    N = np.column_stack((u_x, u_y, u_z))

    R = R_between_coordinate_system(N, M)
    return R

def caculate_world_plane(points:list, world_origin:SE3):
    assert len(points) >= 3, "At least 3 points are required."
    assert len(points) == 4, "Only 4 points are supported."
    
    origin = points[0]
    point_p, point_q, point_r = points[1:]
    
    # Calculate plane equation
    # a, b, c, d = caculate_plan_equation_coefficients(origin, [point_p, point_q, point_r])
    # plan_normal = np.array([a, b, c])
    # plan_normal = plan_normal / np.linalg.norm(plan_normal)
    
    
    v1 = point_p - origin
    v2 = point_q - origin
    vn = np.cross(v1.T, v2.T).T
    
    u_x = v1 / np.linalg.norm(v1)
    u_z = vn / np.linalg.norm(vn)
    u_y = np.cross(u_z.T, u_x.T).T
    
    t = origin - world_origin
    R = R_metrix(u_x, u_y, u_z)
    
    return R, t

def paper2base(paper_coordinate, R, t):
    paper_coordinate = SE3(paper_coordinate)
    base_coordinate = R * paper_coordinate + t
    return base_coordinate

def distance_between_2d_points(point_a, point_b):
    distance = np.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)
    return distance

class Draw2dDxf(Task):
    def __init__(self):#, name=None, description=None, func=None):
        # super().__init__(name=name, description=description, func=func)
        self._robot_arm_ready = False
        self._camera_ready = False
        
    
    def _hardware_setup(self):
        try:
            camera_setup()
            logging.info("Camera setup.")
            logging.info("Camera setup.")
            robot_arm_setup()
            self._camera_ready = True
            logging.info("Robot arm setup.")
        except Exception as e:
            logging.error("Hardware setup failed.")
            logging.error(e)
            self._hardware_shotdown()
            raise e
        
    def _hardware_shotdown(self):
        logging.info("Shutting down hardware...")
        if self._robot_arm_ready:
            robot_arm_shotdown()
            logging.info("Robot arm shotdown.")
        if self._camera_ready:
            robot_arm_shotdown()
            logging.info("Camera shotdown.")
    
    
    
    def __del__(self):
        logging.info("Deleting Draw2dDxf instance...")
        self._hardware_shotdown()
        
    def __call__(self, dxf_file_path, intrinsic_matrix, extrinsic_matrix, distortion_coefficients):
        self._hardware_setup()
        image, depth = camera_capture()
        
        paper_corners_image_coordinate = detect_paper()
        paper_corners_camera_corrdinate = [image2camera(u, v, depth, intrinsic_matrix) for u, v, depth in paper_corners_image_coordinate]
        paper_corners_world_corrdinate = [camera2world(x, y, z, camera2world_matrix) for x, y, z in paper_corners_camera_corrdinate]
        world_origin = paper_corners_world_corrdinate[0]
        other_points = paper_corners_world_corrdinate[1:]
        R, t = caculate_world_plane(other_points, world_origin)
                
        paths = parse_dxf(dxf_file_path)
        paths_base_frame = []
        
        for path in paths:
            draw_point_base_frame = [paper2base(paper_coordinate, R, t) for paper_coordinate in path]
            paths_base_frame.append(draw_point_base_frame)
            
            
        continue_draw_list = [True] * len(paths_base_frame)
        continue_draw_threshold = 0.1
        
        if len(paths_base_frame) > 1:
            for i in range(paths_base_frame[0] - 1):
                d1 = distance_between_2d_points(paths_base_frame[0][i], paths_base_frame[0][i + 1])
                d2 = distance_between_2d_points(paths_base_frame[1][i], paths_base_frame[1][i + 1])
                if d1 > d2:
                    paths_base_frame[i + 1].reverse()
                if min(d1, d2) > continue_draw_threshold:
                    continue_draw_list[i] = False
        
        for path in enumerate(paths_base_frame):
            for point in path:
                # move to point
            if continue_draw_list[path] == False:
                # lift up
        self._hardware_shotdown()
        
if __name__ == '__main__':
    pass