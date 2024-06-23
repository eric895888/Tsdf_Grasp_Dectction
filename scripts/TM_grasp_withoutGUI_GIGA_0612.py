#!/usr/bin/env python


import rospy
import cv2
import cv_bridge
import json
import logging
from pathlib import Path
from robotic_drawing.control.robot_arm.yaskawa.mh5l import MH5L
from robotic_drawing.control.tool.robotiq.gripper_2f85 import Gripper2F85
import numpy as np
import argparse
from pathlib import Path
import sensor_msgs.msg

from vgn import vis
from vgn.experiments.clutter_removal import State
from vgn.detection import VGN
from vgn.perception import *
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform
import os
import sys

import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
from vgn.detection_implicit import VGNImplicit
import open3d as o3d
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
sys.path.append(os.path.dirname(__file__) + os.sep)  #os.sep 表示当前操作系统的路径分隔符，如在 Windows 上是 \，在 Unix 上是 /。

#TODO手臂連線參數
HOST = "192.168.255.10" 
PORT = 11000
SPEED = 400
#TODO Joint位置 T + 52680 after install bin on flang
INIT_JOINT_CONFIGURE = [
    -126330,
    -18970,
    14748,
    -1553,
    -101441,
    46069 + 52680
]

Place_JOINT_CONFIGURE = [
    -52956,
    21581,
    -627,
    967,
    -65817,
    70995
]

#TODO90度的旋轉矩陣 以及下爪深度
offset = 70 #表示z軸向下爪幾公分單位是mm
theta = np.radians(-90)
T_gripper_Rz_90 = np.array([
    [np.cos(theta), -np.sin(theta), 0, 0],
    [np.sin(theta), np.cos(theta), 0, 0],
    [0, 0, 1, offset],
    [0, 0, 0, 1]
])

#TODO校正參數
CALIBRATION_PARAMS_SAVE_DIR = Path(__file__).parent / "calibration_params/2024_06_13" 


#TODO表示相機座標到aruco的座標轉換
ArUco_json_filename = "./scripts/quaternion.json"
with open(ArUco_json_filename, 'r') as f:
    loaded_data = json.load(f)
tvec = loaded_data["tvec"]
quaternion = loaded_data["quaternion"]
print(tvec)
print(quaternion)
#T_cam_task_m = Transform(Rotation.from_quat([0.0091755 ,  0.9995211 ,  0.00176319 ,-0.02950025]), [ 0.16363484, -0.14483834 , 0.44753983])
T_cam_task_m = Transform(Rotation.from_quat(quaternion), tvec)
#T_cam_task_m = Transform(Rotation.from_quat([0.01490109 , 0.96916517 ,-0.02853229 , 0.24430051]), [0.08855362, -0.14576081 , 0.48201059])#能正常運作的


class PandaGraspController(object):
    def __init__(self, args):
        
        self.finger_depth=0.05
        self.size = 0.3 #workspace空間
        self.tsdf_server = TSDFServer()
        self.plan_grasps = VGNImplicit(model_path=args.model,
                                       model_type="giga", 
                                       best=True, 
                                       force_detection=True,
                                      )
        
        rospy.loginfo("Ready to take action")

    def run(self, robot_arm,T_cam2gripper, gripper):
        vis.clear()
        vis.draw_workspace(self.size)
        logging.info("[Robot] Move to initial pose by joint configure")
        ###
        
        tsdf, pc = self.acquire_tsdf()
        
        
        vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(np.asarray(pc.points))
        rospy.loginfo("Reconstructed scene")

        state = State(tsdf, pc) #模擬環境跑一次看看
        grasps, scores, planning_time= self.plan_grasps(state)
        #vis.draw_grasps(grasps, scores, self.finger_depth)
       
        rospy.loginfo("Planned grasps")

        if len(grasps) == 0:
            rospy.loginfo("No grasps detected")
            return

        grasp, score = self.select_grasp(grasps, scores)
        vis.draw_grasp(grasp, score, self.finger_depth)
        rospy.loginfo("Selected grasp")
        
        label = self.execute_grasp(grasp, robot_arm=robot_arm, T_Gripper_Camera=T_cam2gripper, gripper=gripper)
        rospy.loginfo("Grasp execution")
        rospy.sleep(1)
        logging.info("[Robot] Move to initial pose by joint configure")

    def acquire_tsdf(self): #移動去其他視角拍照
        self.tsdf_server.reset()
        self.tsdf_server.integrate = True
        
        # #目前先拍一張就好因此註解掉
        # for Pose in iter(fr): #拍照用
        #     #QApplication.processEvents()
        #     Pose = Pose.replace('[', '').replace(']', '').replace('\n', '').split(', ')
        #     #print(Pose)
        #     Pose = np.array(Pose, dtype="float")
        #     print('Position:' + str(Pose)) 
        #     self.robot.set_TMPos([Pose[0],Pose[1],Pose[2], Pose[3], Pose[4], Pose[5]])  
        rospy.sleep(3.0)
        self.tsdf_server.integrate = False
        tsdf = self.tsdf_server.low_res_tsdf
        pc = self.tsdf_server.high_res_tsdf.get_cloud()

        #TODO 顯示mesh 可以看到平滑表面
        # mesh = self.tsdf_server.high_res_tsdf._volume.extract_triangle_mesh()
        # mesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh])

        print("-----------------")
        print(pc)  
        print("-----------------")
        return tsdf, pc

    def select_grasp(self, grasps, scores):
        # select the highest grasp
        heights = np.empty(len(grasps))
        for i, grasp in enumerate(grasps):
            heights[i] = grasp.pose.translation[2]
        idx = np.argmax(heights)
        grasp, score = grasps[idx], scores[idx]

        # make sure camera is pointing forward
        rot = grasp.pose.rotation
        axis = rot.as_matrix()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        return grasp, score

    def execute_grasp(self, grasp, robot_arm, T_Gripper_Camera, gripper):  #夾取用的座標

        T_task_grasp = grasp.pose
        print("模型測出來的位置")
        print(T_task_grasp.translation)
        #T_base_grasp = self.T_base_task
        
        print("grasp tsdf位置",grasp)
        T_Base_Grasp=self.EstimateCoord(T_task_grasp, T_Gripper_Camera, robot_arm)
  

        robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
        pregrasp_offset= -50 #mm
        #移動到前方
        T_Grasp_Pregrasp = np.array([ 
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, pregrasp_offset],
            [0, 0, 0, 1]
        ])
        T_Base_Pregrasp = T_Base_Grasp @ T_Grasp_Pregrasp
        #T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])  #移動到前方
        #T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])   #夾起來後往後

        # T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        # T_base_retreat = T_base_grasp * T_grasp_retreat
        #移動到指定位置
        #或是直接使用T_base_grasp
        # point=T_base_pregrasp * T_tcp_tool0
        
        #degree=T_base_grasp.rotation.as_euler('xyz', degrees=False)

        # t_task_grasp = T_task_grasp.translation
        # R_task_grasp = T_task_grasp.rotation

        # TODO: 夾取座標後退
        Base_point_t = T_Base_Pregrasp[:3, 3] #3x1 T
        Base_point_r = T_Base_Pregrasp[:3,:3] #3x3
        Base_R_degree= Rotation.from_matrix(Base_point_r).as_euler('xyz',degrees=True)
        robot_arm.move_to_pose(
            Tx=Base_point_t[0],
            Ty=Base_point_t[1],
            Tz=Base_point_t[2],
            Rx=Base_R_degree[0],
            Ry=Base_R_degree[1],
            Rz=Base_R_degree[2],
            speed=SPEED)
        
        ## TODO: 夾取座標
        Base_point_t = T_Base_Grasp[:3, 3] #3x1 T
        Base_point_r = T_Base_Grasp[:3,:3] #3x3
        Base_R_degree= Rotation.from_matrix(Base_point_r).as_euler('xyz',degrees=True)
        robot_arm.move_to_pose(
            Tx=Base_point_t[0],
            Ty=Base_point_t[1],
            Tz=Base_point_t[2],
            Rx=Base_R_degree[0],
            Ry=Base_R_degree[1],
            Rz=Base_R_degree[2],
            speed=SPEED)
        #TODO: gripper
        gripper.close()
        time.sleep(3)

        # TODO: 夾取座標後退
        Base_point_t = T_Base_Pregrasp[:3, 3] #3x1 T
        Base_point_r = T_Base_Pregrasp[:3,:3] #3x3
        Base_R_degree= Rotation.from_matrix(Base_point_r).as_euler('xyz',degrees=True)
        robot_arm.move_to_pose(
            Tx=Base_point_t[0],
            Ty=Base_point_t[1],
            Tz=Base_point_t[2],
            Rx=Base_R_degree[0],
            Ry=Base_R_degree[1],
            Rz=Base_R_degree[2],
            speed=SPEED)
        
        
        robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)  #初始位置
        robot_arm.move_to_joint_config(Place_JOINT_CONFIGURE, SPEED) #擺放位置
        gripper.on()
        robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED) #初始位置

    def EstimateCoord(self,T_task_grasp, T_Gripper_Camera, robot_arm):  #計算座標
        tcp_pose = robot_arm.controller.get_robot_pose(tool_num=0) #讀取手臂本身座標
        #tcp_pose = [88.24895477294922, -567.9349365234375, 362.3840637207031,-177.87, -4.29, -80.39]  #範例座標位置
        print(f"TCP POSE: {tcp_pose}")
        tvec = tcp_pose[:3]

        x_angle, y_angle, z_angle = tcp_pose[3:]
        # rvec = Rotation.from_euler(
        #     "xyz", [x_angle, y_angle, z_angle], degrees=True
        # ).as_rotvec(degrees=False)
        
        T_Base_Gripper= np.eye(4)
        T_Base_Gripper[:3, :3] = Rotation.from_euler(
            "xyz", [x_angle, y_angle, z_angle], degrees=True
        ).as_matrix()
        T_Base_Gripper[:3, 3] =np.array(tvec).T

        print(f"TCP POSE mat: {T_Base_Gripper}")

        print("---------")
        print(T_cam_task_m.as_matrix())
        Task2Camera_r = T_cam_task_m.as_matrix()[:3,:3]
        Task2Camera_t = T_cam_task_m.as_matrix()[:3,3]  #公尺
        print(Task2Camera_r)
        print(Task2Camera_t*1000)

        T_Camera_Task=np.r_[np.c_[Task2Camera_r, Task2Camera_t*1000], [[0, 0, 0, 1]]]
        
        T_Task_Grasp = np.eye(4)
        T_task_grasp_t = T_task_grasp.as_matrix()[:3,3]
        T_task_grasp_r = T_task_grasp.as_matrix()[:3,:3]
        T_task_grasp_Rx_deg, T_task_grasp_Ry_deg, T_task_grasp_Rz_deg = Rotation.from_matrix(T_task_grasp_r).as_euler('xyz', degrees=True)
        print("原始角度",T_task_grasp_Rx_deg, T_task_grasp_Ry_deg, T_task_grasp_Rz_deg )
        
        T_Task_Grasp[:3,:3] = T_task_grasp_r
        T_Task_Grasp[:3,3] = T_task_grasp_t * 1000

        # T_Task_Grasp=np.r_[np.c_[T_task_grasp_r, T_task_grasp_t], [[0, 0, 0, 1]]]
        print("grasp2task",T_Task_Grasp)

        #TODO顯示各個座標系
        tm = TransformManager()
        tm.add_transform("grasp", "task", T_Task_Grasp)
        tm.add_transform("gripper", "robot", T_Base_Gripper)
        tm.add_transform("camera", "gripper", T_Gripper_Camera)
        tm.add_transform("task", "camera", T_Camera_Task)
        ax = tm.plot_frames_in("task", s=100)
        ax.set_xlim((-1000, 1000))
        ax.set_ylim((-1000, 1000))
        ax.set_zlim((-1000, 1000))
        #plt.show() 顯示座標圖
        
        
        #TODO base座標計算從右邊往左看,相機座標到夾爪座標再到base座標
        T_Base_Grasp = T_Base_Gripper @ T_Gripper_Camera @ T_Camera_Task @ T_Task_Grasp @ T_gripper_Rz_90    
        print("T_Gripper_Camera", T_Gripper_Camera)
        print("T_Base_Gripper",T_Base_Gripper)
        print("T_Camera_Task",T_Camera_Task)
        
        return T_Base_Grasp


class TSDFServer(object):
    def __init__(self):
        self.cam_frame_id = "camera_depth_optical_frame"
        self.cam_topic_name = "/camera/depth/image_rect_raw"
        self.intrinsic = CameraIntrinsic(640, 480, 606.77737126, 606.70030146, 321.63287183, 236.95293136)
        self.distortion=np.array([ 4.90913179e-02 , 5.22699002e-01, -2.65209452e-03  ,1.13033224e-03,-2.17133474e+00]) #影像扭曲嚴重時修正使用,若幾乎沒有扭曲可以不用

        self.size = 0.3 #30cmworksapce
        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_tree = ros_utils.TransformTree()
        self.integrate = False
        rospy.Subscriber(self.cam_topic_name, sensor_msgs.msg.Image, self.sensor_cb)
        
    def reset(self):
        self.low_res_tsdf = TSDFVolume(self.size, 40)
        self.high_res_tsdf = TSDFVolume(self.size, 120)

    def sensor_cb(self, msg):
        if not self.integrate:
            return

        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001
        #T_cam_task_m =  ArUco_detection.Img_ArUco_detect(img,self.intrinsic,self.distortion)
        #用來廣播目標平面到相機的關係，也就是相機外參
        self.tf_tree.broadcast_static(
            T_cam_task_m, self.cam_frame_id, "task"
        )


        self.low_res_tsdf.integrate(img, self.intrinsic, T_cam_task_m)
        self.high_res_tsdf.integrate(img, self.intrinsic, T_cam_task_m)


def main(args):
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController(args)

    #TODO 校正參數
    with open(str(CALIBRATION_PARAMS_SAVE_DIR / 'calibration_params.json')) as f:
        calibration_params = json.load(f)
        T_cam2gripper = np.array(calibration_params["T_cam2gripper"])

    #TODO夾爪控制 
    #目前註解掉gripper="" 
    logging.info("Init")
    gripper = Gripper2F85()
    logging.info("Connect")
    success = gripper.connect()
    logging.info("Reset")
    success = gripper.reset()

    #手臂控制 #目前註解掉
    robot_arm = MH5L(host=HOST, port=PORT)
    #robot_arm="" 
    logging.info("[Robot] Init")
    robot_arm.power_on()
    logging.info("[Robot] Power on")
    robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
    logging.info("[Robot] Move to initial pose by joint configure")
    
    # ###
    ### TODO: times
    while(True): 
        #robot_arm.move_to_joint_config([-126330, -19968, 12272, -1538, -100260, 46100 + 52680], SPEED) 
        #robot_arm.move_to_joint_config([-193854,-7245,4297,-24799,-103917,119435], SPEED)#可行
        robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
        panda_grasp.run(robot_arm, T_cam2gripper, gripper) 
        
    
    # ### TODO test
    #robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
    logging.info("[Robot] Move to initial pose by joint configure")

    #robot_arm.power_off() 
    logging.info("[Robot] Power off")
    ### 
    
#TODO 球形座標目前沒用到
def camera_on_sphere(origin, radius, theta, phi):  #球形座標
    eye = np.r_[
        radius * sin(theta) * cos(phi),
        radius * sin(theta) * sin(phi),
        radius * cos(theta),
    ]
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
    return Transform.look_at(eye, target, up) * origin.inverse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default="/home/robotic/Grasp_detection_GIGA/scripts/data/models/TT_cube_giga_epoch20.pt")
    args = parser.parse_args()
    main(args)
