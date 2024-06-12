#!/usr/bin/env python

"""
Open-loop grasp execution using a Panda arm and wrist-mounted RealSense camera.
"""

import argparse
from pathlib import Path

import cv_bridge
import RobotControl_func_ros1 as RobotControl_func
#import franka_msgs.msg
import geometry_msgs.msg
import numpy as np
import rospy
import sensor_msgs.msg

from vgn import vis
from vgn.experiments.clutter_removal import State
from vgn.detection import VGN
from vgn.perception import *
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform
from vgn.utils.panda_control import PandaCommander

import rospy
import RobotControl_func_ros1 as RobotControl_func
from gripper import Gripper 
import shutil
import os
import sys
import ArUco_detection

from vgn.detection import *
from vgn.perception import *
sys.path.append(os.path.dirname(__file__) + os.sep)  #os.sep 表示当前操作系统的路径分隔符，如在 Windows 上是 \，在 Unix 上是 /。

# tag lies on the table in the center of the workspace
#238.73208618164062, -399.728271484375, 33.843978881835945 mm
T_base_tag = Transform(Rotation.identity(), [0.2387, -0.3997, 0.033]) #手臂下去點就知道位置了!!!!!!!!!! 單位是m
round_id = 0
#[289.77935791015625, -464.6210632324219, 296.10107421875, 3.0886689848271933, 0.045914536701512146, 0.7913491300112848]
#置中座標
# [[ 0.99959658 -0.00447668 -0.02804718]
#  [ 0.00441263  0.99998751 -0.00234523]
#  [ 0.02805733  0.00222052  0.99960385]]
# [[-85.23109251]
#  [-73.48531014]
#  [355.95850582]]  目前位置的相機外參

class PandaGraspController(object):
    def __init__(self, args):
        self.robot_error = False

        self.base_frame_id = "panda_link0" #用途不明
        self.tool0_frame_id = "panda_link8" #用途不明

        
        #self.finger_depth = rospy.geT_tool0_tcpt_param("~finger_depth") 這是從.yaml的設定檔案拉參數出來用的方式
        self.finger_depth = 0.04 # 單位m

        #self.size = 6.0 * self.finger_depth
        self.size = 0.3

        self.robot = RobotControl_func.RobotControl_Func() #手臂控制
        self.grip = Gripper()

        #self.setup_panda_control()
        self.tf_tree = ros_utils.TransformTree()
        self.define_workspace()
        self.create_planning_scene()
        self.tsdf_server = TSDFServer()
        self.plan_grasps = VGN(args.model, rviz=True)

        rospy.loginfo("Ready to take action")

    def setup_panda_control(self):
        # rospy.Subscriber(
        #     "/franka_state_controller/franka_states",
        #     franka_msgs.msg.FrankaState,
        #     self.robot_state_cb,
        #     queue_size=1,
        # )
        rospy.Subscriber(
            "/joint_states", sensor_msgs.msg.JointState, self.joints_cb, queue_size=1
        )
        self.pc = PandaCommander()
        self.pc.move_group.set_end_effector_link(self.tool0_frame_id)

        
    def define_workspace(self):
        z_offset = -0.06
        t_tag_task = np.r_[[-0.5 * self.size, -0.5 * self.size, z_offset]]
        T_tag_task = Transform(Rotation.identity(), t_tag_task)
        self.T_base_task = T_base_tag * T_tag_task  #物體到base的變換矩陣？？

        self.tf_tree.broadcast_static(self.T_base_task, self.base_frame_id, "task")  #檢查這裡
        rospy.sleep(1.0)  # wait for the TF to be broadcasted

    def create_planning_scene(self):
        # collision box for table
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame_id
        msg.pose = ros_utils.to_pose_msg(T_base_tag)
        msg.pose.position.z -= 0.01  #?未知
        #下方式避障用
        #self.pc.scene.add_box("table", msg, size=(0.6, 0.6, 0.02))

        rospy.sleep(1.0)  # wait for the scene to be updated

    def robot_state_cb(self, msg):
        detected_error = False
        if np.any(msg.cartesian_collision):
            detected_error = True
        # for s in franka_msgs.msg.Errors.__slots__:
        #     if getattr(msg.current_errors, s):
        #         detected_error = True
        if not self.robot_error and detected_error:
            self.robot_error = True
            rospy.logwarn("Detected robot error")

    def joints_cb(self, msg): #?
        self.gripper_width = msg.position[7] + msg.position[8]

    def run(self):
        vis.clear()
        vis.draw_workspace(self.size)
        #self.pc.move_gripper(0.08)

        #self.pc.home()
        self.robot.set_TMPos([289.77935791015625, -464.6210632324219, 296.10107421875, 3.0886695174594117, 0.045914965304000084, 0.7913568531784461])

        tsdf, pc = self.acquire_tsdf()
        vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(np.asarray(pc.points))
        rospy.loginfo("Reconstructed scene")

        state = State(tsdf, pc)
        grasps, scores, planning_time = self.plan_grasps(state)
        vis.draw_grasps(grasps, scores, self.finger_depth)
        rospy.loginfo("Planned grasps")

        if len(grasps) == 0:
            rospy.loginfo("No grasps detected")
            return

        grasp, score = self.select_grasp(grasps, scores)
        vis.draw_grasp(grasp, score, self.finger_depth)
        rospy.loginfo("Selected grasp")

        # self.pc.home()
        label = self.execute_grasp(grasp)
        # rospy.loginfo("Grasp execution")

        # if self.robot_error:
        #     self.recover_robot()
        #     return

        # if label:
        #     self.drop()
        # self.pc.home()
        #self.robot.set_TMPos([289.77935791015625, -464.6210632324219, 296.10107421875, 3.0886695174594117, 0.045914965304000084, 0.7913568531784461]) #home
        

    def acquire_tsdf(self): #移動去其他視角拍照
        fr = open("/media/eric/Disk/vgn/scripts/PosSet.txt", 'r+')  #這時還沒有抓取joint的function所以使用position當作點位
        
        #print(Total_pose)
        # print(j)
        self.tsdf_server.reset()
        self.tsdf_server.integrate = True
        
        # #先拍一張就好
        # for Pose in iter(fr): #拍照用
        #     #QApplication.processEvents()
        #     Pose = Pose.replace('[', '').replace(']', '').replace('\n', '').split(', ')
        #     #print(Pose)
        #     Pose = np.array(Pose, dtype="float")
        #     print('Position:' + str(Pose)) 
        #     self.robot.set_TMPos([Pose[0],Pose[1],Pose[2], Pose[3], Pose[4], Pose[5]])  

        self.tsdf_server.integrate = False
        tsdf = self.tsdf_server.low_res_tsdf
        pc = self.tsdf_server.high_res_tsdf.get_cloud()
        # #測試中＃＃＃＃
        # vis.clear()
        # vis.draw_workspace(0.3)  #30cm
        # from pathlib import Path
        # str_path = "/media/eric/Disk/vgn/scripts/data/models/vgn_conv.pth"
        # path = Path(str_path)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.net = load_network(path, self.device)
        # tic = time.time()
        # tsdf_vol = tsdf.get_grid()
        # voxel_size = tsdf.voxel_size
        # print("Extract grid  ", time.time() - tic)

        # tic = time.time()
        # qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        # print("Forward pass   ", time.time() - tic)

        # tic = time.time()
        # qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)
        # print("Filter         ", time.time() - tic)

        # vis.draw_quality(qual_vol, voxel_size, threshold=0.01)
        # #測試中＃＃＃＃

        print("-----------------")
        print(pc)  #目前0點
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

    def execute_grasp(self, grasp):  #夾取用的 grasp是從world到gripper座標

        #這一項或許不用，取決於到底是以夾爪末端還是法蘭面當作工具座標
        T_tool0_tcp = Transform(Rotation.identity(),[0.00065, 0.00331, 0.187]) # TODO   #單位是m 工法蘭面距離tool0到工具組中心tcp的關係  使用tmflow從工具組中心點值中曲出來的
        #T_tool0_tcp = Transform.from_dict(rospy.get_param("~T_tool0_tcp"))  # TODO   
        T_tcp_tool0 = self.T_tool0_tcp.inverse()

        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])  #移動到前方
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])   #夾起來後往後

        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        T_base_retreat = T_base_grasp * T_grasp_retreat
        print("有使用tcp_tool0")
        print(T_base_pregrasp * T_tcp_tool0)
        #移動到指定位置
        self.robot.set_TMPos(T_base_pregrasp * T_tcp_tool0)  #??

        #self.pc.goto_pose(T_base_pregrasp * T_tcp_tool0, velocity_scaling=0.2)
        self.approach_grasp(T_base_grasp)

        if self.robot_error:
            return False

        self.pc.grasp(width=0.0, force=20.0)

        if self.robot_error:
            return False

        self.pc.goto_pose(T_base_retreat * T_tcp_tool0)

        # lift hand
        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        self.pc.goto_pose(T_base_lift * T_tcp_tool0)

        if self.gripper_width > 0.004:
            return True
        else:
            return False

    def approach_grasp(self, T_base_grasp):
        self.pc.goto_pose(T_base_grasp * self.T_tcp_tool0)

    def drop(self):  #放置位置
        self.pc.goto_joints(
            [0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931], 0.2, 0.2
        )
        self.pc.move_gripper(0.08)


class TSDFServer(object):
    def __init__(self):
        self.cam_frame_id = "camera_depth_optical_frame"
        self.cam_topic_name = "/camera/depth/image_rect_raw"
        self.intrinsic = CameraIntrinsic(640, 480, 606.77737126, 606.70030146, 321.63287183, 236.95293136)
        #self.size = 6.0 * rospy.get_param("~finger_depth") #finger_depth  但是6的意義不明 但我們的爪子應該是0.04 0.05是paper預設大小
        self.size = 0.3 #finger_depth  但是6的意義不明 但我們的爪子應該是0.04 0.05是paper預設大小
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
        # T_cam_task = self.tf_tree.lookup(    #可能要自己抓相機外參
        #     self.cam_frame_id, "task", msg.header.stamp, rospy.Duration(0.1)
        # )
        
        T_cam_task = Transform(  #目前的初始位置,偏移矩陣單位是m
            Rotation.from_quat([ 0.99989849,  0.00114155, -0.01402755,  0.00222255]), [-0.08523, -0.07348 , 0.35595]  #我們的
            
        )
        # broadcast the tf tree (for visualization)
        #用來廣播目標平面到相機的關係，也就是相機外參
        self.tf_tree.broadcast_static(
            T_cam_task, self.cam_frame_id, "task"
        )

        self.low_res_tsdf.integrate(img, self.intrinsic, T_cam_task)
        self.high_res_tsdf.integrate(img, self.intrinsic, T_cam_task)


def main(args):
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController(args)
    
    while True:
        panda_grasp.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model", type=Path, default="/home/eric/catkin_ws/src/vgn/scripts/data/models/vgn_conv.pth")
    args = parser.parse_args()
    main(args)
