import rospy
import cv2
import cv_bridge
import json
import logging
from pathlib import Path
from robotic_drawing.control.robot_arm.yaskawa.mh5l import MH5L
from robotic_drawing.control.tool.robotiq.gripper_2f85 import Gripper2F85
import numpy as np
from ui.UI import Ui_MainWindow
from pathlib import Path
import geometry_msgs.msg
import sensor_msgs.msg
from vgn import vis
from vgn.experiments.clutter_removal import State
from vgn.detection import VGN
from vgn.perception import *
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform
import os
import sys
import ArUco_detection
from vgn.detection import *
from vgn.perception import *
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
import pyrealsense2 as rs
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from vgn.detection_implicit import VGNImplicit
import EIHCali
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH") #用來讓pyqt5插件能動,它用於從系統環境變量中移除名為 QT_QPA_PLATFORM_PLUGIN_PATH 的變量
sys.path.append(os.path.dirname(__file__) + os.sep)  #os.sep 表示当前操作系统的路径分隔符，如在 Windows 上是 \，在 Unix 上是 /。

# TODO 手臂參數
HOST = "192.168.255.10"
PORT = 11000
SPEED = 600
# TODO 初始位置及擺放位置
INIT_JOINT_CONFIGURE = [ #初始位置
    -126330,
    -18970,
    14748,
    -1553,
    -101441,
    46069 + 52680
]
Place_JOINT_CONFIGURE = [ #擺放位置
    -52956,
    21581,
    -627,
    967,
    -65817,
    70995
]

CALIBRATION_PARAMS_SAVE_DIR = Path(__file__).parent / "calibration_params/2024_03_05" #校正參數
##########



#表示相機座標到aruco的座標
T_cam_task_m = Transform(Rotation.from_quat([0.0091755 ,  0.9995211 ,  0.00176319 ,-0.02950025]), [ 0.16363484, -0.14483834 , 0.44753983])
round_id = 0

class RGBDThread(QThread):
    rgbd_trigger = pyqtSignal(object)
    def __init__(self):
        super(RGBDThread, self).__init__()
        self._mutex = QMutex()
        self._running = True
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        profile = self.pipeline.start(self.config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  
        print("Depth units setted: ", '%.5f'%self.depth_scale)  #realsense 400系列預設是0.001
        self.scale = 1
    def __del__(self):
        self.pipeline.stop()
        self.wait()
    def run(self): #去頭去尾 [self.color_image,depth_image_3c,self.depth_image] 代表rgb,三通道的depth_map,原本的depth_map
            try:
                while self.running():
                    # Wait for a coherent pair of frames: depth and color
                    try:
                        frames = self.pipeline.wait_for_frames()
                        aligned_frames = self.align.process(frames)
                        depth_frame = aligned_frames.get_depth_frame()
                        color_frame = aligned_frames.get_color_frame()
                        # color_frame = frames.get_color_frame()
                        # depth_frame = frames.get_depth_frame()
                        if not depth_frame or not color_frame:
                            print("continue")
                            continue
                    except Exception as e:
                        print(str(e))
                        pass
                    self.d_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                    # Convert images to numpy arrays
                    self.depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.uint8(color_frame.get_data())
                    
                    self.color_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
                    #depth_image_3c = cv2.cvtColor(depth_image_3c,cv2.COLOR_BGR2RGB)
                    depth_map_normalized = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #normalize 0~255
                    #height, width = depth_map_normalized.shape
                    
                    self.rgbd_pair = [self.color_image,depth_map_normalized]
                    self.rgbd_trigger.emit(self.rgbd_pair)
                    # cv2.waitKey(5)
            except NameError as e:
                print(e)
                self.pipeline.stop()
    def pause(self):
        print("pause streaming")
        self._mutex.lock()
        self._running = False
        # self.pipeline.stop()
        self._mutex.unlock()
    def restart(self):
        global mode
        mode = 2  #TODO: 停止绘制 BBox和 Point
        print("restart streaming")
        self._mutex.lock()
        self._running = True
        # self.pipeline.start()
        self._mutex.unlock()
        self.start()
    def running(self):
        try:
            self._mutex.lock()
            return self._running
        finally:
            self._mutex.unlock()
    def getDepth(self): 
        try:
            frames = self.pipeline.wait_for_frames()
            depth=self.align.process(frames).get_depth_frame().get_distance(realX, realY)*1000
            return depth
        except Exception as e:
            return e
        
    def get_depth_profile(self):  #來自realsense
        return self.depth_scale, self.d_intrin
    # def normalize(self, depth):
    #     min_val = 30
    #     max_val = 255
    #     d_med = (np.max(depth) + np.min(depth)) / 2.0
    #     d_diff = depth - d_med
    #     depth_rev = np.copy(depth)
    #     depth_rev = d_diff * (-1) + d_med
    #     depth_rev = depth_rev - np.min(depth_rev)

    #     depth_rev = cv2.normalize(depth_rev, None, min_val, max_val, cv2.NORM_MINMAX, cv2.CV_8UC1)
    #     return depth_rev
    #     #return cv2.normalize(depth, None, min_val, max_val, cv2.NORM_MINMAX, cv2.CV_8UC1)

class ArmThread(QThread):
    # control signal: working signal of robotarm
    # Input: position of bounding box
    # output: finish signal
    release_trigger = pyqtSignal(object)
    

    def __init__(self):
        super(ArmThread,self).__init__()
        self._mutex = QMutex()
        self.grip = Gripper()     #夾爪控制
        self.pose = None
        self._running = True
        self.release = False
        self.robot = RobotControl_func.RobotControl_Func() #手臂控制

    def initArmGripper(self):
        # to init arm position
        # init gripper
        self.grip.gripper_reset()

    def run(self):
        while self.running():
            if self.pose == None:
                #print("don't move")
                continue
            else:
                self.pick(self.pose)
                self.pose = None

    def testArm(self): #
        position = self.robot.get_TMPos() #[x,y,z,u,v,w]
        print("pos=",position)
        return position

    def calPosXY(self,camera_point):
        # print(camera_point)
        camera_xy = [camera_point[0],camera_point[1]]
        camera_xy.append (1.0)
        # print(camera_point)
        arm_point = np.array (np.array(camera_xy)) @ self.trans_mat
        print ("arm_point",arm_point)
        return arm_point

    def pick(self,camera_point): #?
        print (camera_point,type (camera_point))
        arm_point = self.calPosXY (camera_point)
        pos = self.cntrl.get_robot_pos ()
        # pos = self.cntrl.robot_info
        new_point = arm_point
        # print(int(new_point[0]*1000), int(new_point[1]*1000), pos[2], pos[3], pos[4], pos[5])
        self.cntrl.move_robot_pos (int (new_point[0] * 1000),int (new_point[1] * 1000),pos[2],pos[3],pos[4],pos[5],2000)
        # self.cntrl.move_robot_pos(-339197, -264430, 156320, -1668426, -24203, -74088, 1000)
        self.cntrl.wait_move_end ()
        depth_diff = (self.baseline_depth - camera_point[2])
        arm_z = 10000 + round (40 * depth_diff)
        if arm_z > 156320:
            arm_z = str (156320)
        elif arm_z < 10000:
            arm_z = str (10000)
        else:
            arm_z = str (arm_z)
        # print(10000+round(40*depth_diff))

        self.cntrl.move_robot_pos (int (new_point[0] * 1000),int (new_point[1] * 1000),arm_z,pos[3],pos[4],pos[5],2000)
        self.cntrl.wait_move_end ()
        self.grip.gripper_off()
        time.sleep(0.5)
        self.cntrl.move_robot_pos(int (new_point[0] * 1000),int (new_point[1] * 1000),pos[2],pos[3],pos[4],pos[5],2000)
        self.cntrl.wait_move_end()
        self.cntrl.move_robot_pos ('-271077','-415768','156320','-1709954','-1907','-104123',2000)
        self.cntrl.wait_move_end ()
        self.grip.gripper_on ()
        time.sleep(0.3)

        print("here")
        # go home
        self.cntrl.move_robot_pos ('2883','-246016','166040','-1709973','-1929','-104740',2000)
        self.cntrl.wait_move_end ()
        self.release = False
        print("fin")
        self.release_trigger.emit (True)

    def goHome(self): #回正
        #135度 self.robot.set_TMPos([100.46757778617632, 546.3748799610293, 405.3811340332031, 3.069622855659274, 0, 0.7746395918047735])
        #原始座標
        self.robot.set_TMPos([315.30413818359375, -457.3872680664063, 405.3842163085938, 3.0696167303887667, -1.108274100572733e-06, 0.7746365957485472])
    def reGrasp(self):
        # print("restart streaming")
        self._mutex.lock ()
        self._running = True
        self._mutex.unlock ()
        self.run ()
    def stopGrasp(self):
        self.goHome()
        self._mutex.lock()
        self._running = False
        self._mutex.unlock()

    def running(self):
        try:
            self._mutex.lock()
            return self._running
        finally:
            self._mutex.unlock()

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        #self._stream_thread = RGBDThread()
        #self._stream_thread.rgbd_trigger.connect(self.updateRGBDFrame)  #在這裡改成用ros同步刷新
        self.RGBFrame.mousePressEvent = self.mouseClicked
        #self.RGBFrame.mouseReleaseEvent = self.mouseRelease
        #--------------------Bind UI with func（Start）--------------------------------
        self.CameraOn.clicked.connect(self.Start2Stop) #连接相机
        self.CameraCalibration.clicked.connect(self.CameraCalibrationF) #Camera Calibration
        self.SaveImages.clicked.connect(self.SaveImagesF)  # Save Image
        self.GripperOpen.clicked.connect(self.GripperOpenF) #Open Gripper
        self.GripperClose.clicked.connect(self.GripperCloseF)  # Close Gripper
        self.ConnectRobtoArm.clicked.connect(self.ConnectRobotArmF) #connect RobotArm
        self.SetInitPos.clicked.connect(self.SetInitPosF) #Set Init Position
        self.GetPos.clicked.connect(self.GetPosF) #Get Current Position
        #self.AutoEIHCalib.clicked.connect(self.AutoEIHCalibF)
        #self.TestCalibration.clicke#onF)
        #self.FindPlane.clicked.connect(self.FindPlaneF)
        #self.PenTouchTest.clicked.connect(self.PenTouchTestF)
        #self.DepthTest.clicked.connect(self.getDepthF)
        # --------------------parameter------------------------------------------------
        self.cali_img_id=1 #相機校正影像編號
        self.intrinsic = np.empty((3,3))
        self.distCoeffs = np.empty((1,5))
        self.Camera2Gripper = np.empty((4,4))
        self.depth_value=0
        # --------------------Bind UI with func（End）--------------------------------
        # --------------------rviz set（start）--------------------------------
        self.base_frame_id = "panda_link0" #用途不明
        self.tool0_frame_id = "panda_link8" #用途不明
        self.size = 0.3 #workspace空間


        #self.robot = RobotControl_func.RobotControl_Func() #手臂控制
        ### TODO tc-huang
        #self.grip = Gripper()
        ###

        #self.setup_panda_control()
        self.tf_tree = ros_utils.TransformTree()
        #self.tsdf_server = TSDFServer()
        #self.plan_grasps = VGN("./scripts/data/models ", rviz=True)
        # --------------------rviz set（End）--------------------------------
    def getDepthF(self):  #取得深度
        self.depth_value=self._stream_thread.getDepth()
        
        self.OutPut.setText("Depth: "+str('%.2f'%self.depth_value)+"mm")
    def FindPlaneF(self):  #??
        # self.FindPlane.setText("9")
        if(self.FindPlane.text() == "Step 1: Find Plane"):
            self.FindPlane.setText("1")
            self._arm_thread.cntrl.get_robot_pos()
            self.OutPut.setText(str(np.array(self._arm_thread.cntrl.robot_info)))
        else:
            if(self.FindPlane.text() != "9"):
                self.FindPlane.setText(str(int(self.FindPlane.text())+1))
                self._arm_thread.cntrl.get_robot_pos()
                self.OutPut.append(str(np.array(self._arm_thread.cntrl.robot_info)))
            else:
                self.FindPlane.setText("Step 1: Find Plane")
                self.OutPut.append("------------\nFinished")

    def TestCalibrationF(self): #抓取點位並且復原 ？？
        # #TODO: Gripper grasp Plane
        # fr = open(sys.path[0] + "/calibration/PosSet.txt", 'r+')
        # dic = eval(fr.read())
        # fr.close()
        # GetPlanePos = StrToArray.StrToArray(dic['GetPlanePos'])
        # ObjectPos = StrToArray.StrToArray(dic['CaliPos'])
        # self._arm_thread.cntrl.move_robot_pos(GetPlanePos[0], GetPlanePos[1], GetPlanePos[2], GetPlanePos[3],
        #                                       GetPlanePos[4], GetPlanePos[5], 2000)
        # self._arm_thread.cntrl.wait_move_end()
        # self.GripperCloseF()
        # time.sleep(1)  # time for Gripper Close
        # #TODO: Make 3 move
        # self._arm_thread.cntrl.move_robot_pos(ObjectPos[0][0], ObjectPos[0][1], ObjectPos[0][2], ObjectPos[0][3], ObjectPos[0][4], ObjectPos[0][5], 500)
        # self._arm_thread.cntrl.wait_move_end()
        # imageInFrame = self._rgb_image
        # image = self.convertQImageToMat(imageInFrame)
        # cv2.imshow("ETH Calibration Test",image)
        # cv2.waitKey(50)
        # #TODO: 不知道什么问题，但是要用waitkey刷新一下RGB Frame
        # imageInFrame = self._rgb_image
        # image = self.convertQImageToMat(imageInFrame)
        # cv2.imshow("ETH Calibration Test", image)
        # cv2.imwrite(sys.path[0] + "/calibration/EyeToHandCali/Test1.jpg", image)
        # cv2.waitKey(50)

        # self._arm_thread.cntrl.move_robot_pos(ObjectPos[5][0], ObjectPos[5][1], ObjectPos[5][2], ObjectPos[5][3],
        #                                       ObjectPos[5][4], ObjectPos[5][5], 500)
        # self._arm_thread.cntrl.wait_move_end()
        # imageInFrame = self._rgb_image
        # image = self.convertQImageToMat(imageInFrame)
        # cv2.imshow("ETH Calibration Test", image)
        # cv2.waitKey(50)
        # #TODO: 不知道什么问题，但是要用waitkey刷新一下RGB Frame
        # imageInFrame = self._rgb_image
        # image = self.convertQImageToMat(imageInFrame)
        # cv2.imshow("ETH Calibration Test", image)
        # cv2.imwrite(sys.path[0] + "/calibration/EyeToHandCali/Test2.jpg", image)
        # cv2.waitKey(50)

        # self._arm_thread.cntrl.move_robot_pos(ObjectPos[8][0], ObjectPos[8][1], ObjectPos[8][2], ObjectPos[8][3],
        #                                       ObjectPos[8][4], ObjectPos[8][5], 500)
        # self._arm_thread.cntrl.wait_move_end()
        # imageInFrame = self._rgb_image
        # image = self.convertQImageToMat(imageInFrame)
        # cv2.imshow("ETH Calibration Test", image)[0.5940838069345818, -0.1385408801148759, 0.46743569946289065, -3.0309942366798928, 0.028533363832244956, 0.7023456205905642]
        # cv2.waitKey(50)
        # #TODO: 不知道什么问题，但是要用waitkey刷新一下RGB Frame
        # imageInFrame = self._rgb_image
        # image = self.convertQImageToMat(imageInFrame)
        # cv2.imshow("ETH Calibration Test", image)
        # cv2.imwrite(sys.path[0] + "/calibration/EyeToHandCali/Test3.jpg", image)
        # cv2.destroyAllWindows()
        # #TODO:Calculate the result
        # EIHCali.TestETHCali()
        # #TODO: Put Chessboard Back
        # self._arm_thread.cntrl.move_robot_pos(GetPlanePos[0], GetPlanePos[1], GetPlanePos[2], GetPlanePos[3],
        #                                       GetPlanePos[4], GetPlanePos[5], 2000)
        # self._arm_thread.cntrl.wait_move_end()[0.5940838069345818, -0.1385408801148759, 0.46743569946289065, -3.0309942366798928, 0.028533363832244956, 0.7023456205905642]
        # self.GripperOpenF()
        # time.sleep(1)  # Time for gripper open
        # self.SetInitPosF()
        pass

    # def DebugCalibF(self): #?
    #     # result = EIHCali.ETHCali()
    #     # self.OutPut.setText("ETH Calibrate Finished\nT:\n" + str(result))
    #     # EIHCali.TestETHCali()
    #     pass
        
    # def AutoEIHCalibF(self): #修正手眼校正中
    #     shutil.rmtree(sys.path[0] + "/Saved_IMG/") #清空資料夾
    #     if not os.path.exists(sys.path[0]+'/Saved_IMG'):
    #         os.makedirs(sys.path[0]+'/Saved_IMG')
    #     fr = open(sys.path[0] + "/EIH_PosSet.txt", 'r+')  #這時還沒有抓取joint的function所以使用position當作點位
    #     #print(Total_pose)
    #     # print(j)
    #     id = 1
    #     Total_poses=[] #用來存成功的位置
    #     for Pose in iter(fr):
    #         #QApplication.processEvents()
    #         Pose = Pose.replace('[', '').replace(']', '').replace('\n', '').split(', ')
    #         #print(Pose)
    #         Pose = np.array(Pose, dtype="float")
    #         print('Position:' + str(Pose)) 
    #         #點位置自己重抓 要轉45度
            
    #         # import math
    #         # cos45 = math.cos(math.radians(45)) #往右轉45度
    #         # sin45 = math.sin(math.radians(45))
    #         # R2 = np.array([[cos45,-sin45,0],[sin45,cos45,0],[0,0,1]])
    #         # ori_xyz = [Pose[0],Pose[1],Pose[2]]
    #         # after_xyz = ori_xyz @ R2    

    #         self._arm_thread.robot.set_TMPos([Pose[0],Pose[1],Pose[2], Pose[3], Pose[4], Pose[5]])  
    #         # print('I am in ' + str(self._arm_thread.testArm()))
            
    #         #在这里刷新一下Frame  不然目前問題出在只會抓取前一pose畫面
    #         cv2.waitKey(300)
    #         self.OutPut.setText("Begin EIH Calibration(NUM" + str(id) + ")")
    #         print("Begin EIH Calibration(NUM " + str(id) + " )")
    #         w = 11
    #         h = 8
    #         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #         # objp = np.zeros((w * h, 3), np.float32)
    #         # objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    #         # # 储存棋盘格角点的世界坐标和图像坐标对
            
    #         #self.updateRGBDFrame(self._rgb_image)
    #         #QApplication.processEvents()
    #         imageInFrame = self._rgb_image
    #         image = self.convertQImageToMat(imageInFrame)
    #         # cv2.imshow("EIH Calibration", image)
    #         # cv2.waitKey(50)
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         # 找到棋盘格角点
    #         ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    #         # 如果找到足够点对，将其存储起来

    #         if ret == True :
    #             Total_poses.append(Pose)
    #             #Total_imgs.append(image) 
    #             cv2.imwrite(sys.path[0] + "/Saved_IMG/" + str(id) + ".png",image)
    #             print("saved:"+str(id)+"\n----------")
    #             sub_corners=cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    #             cv2.drawChessboardCorners(image, (w, h), sub_corners, ret)
    #             cv2.imshow("EIH Calibration", image)
    #             id+=1
    #         else:
    #             print("ERROR:No chessboard was found\n----------BREAK----------")
            
    #         cv2.waitKey(300)
    #         #self.Start2Stop()  #恢復影像
    #         #self.Start2Stop()  #暫停影像
            
    #         cv2.destroyAllWindows()

    #     self.SetInitPosF()
        
    #     #assert(len(Total_poses)>=20) #要大於20張才能繼續 用在gripper到base座標轉換
    #     #assert(len(Total_imgs)>=20)  #要大於20張才能繼續
    #     print(self.intrinsic)
    #     print(self.distCoeffs)
    #     print(sys.path[0])
    #     Task_r_Camera,Task_t_Camera=EIHCali.Find_Extrinsic(sys.path[0],self.intrinsic,self.distCoeffs)   #mtx  <class 'numpy.ndarray'>  ！！！！！！
    #     print(Total_poses)
    #     print(len(Task_r_Camera))
    #     print(len(Task_t_Camera))
    #     print("@@@@@@@@@@@@@@@@@")
    #     self.Camera2Gripper = EIHCali.EyeInHandCalibration(Total_poses,Task_r_Camera,Task_t_Camera)
    #     #result = EIHCali.EyeInHandCalibration(sys.path[0],Total_poses)
    #     self.OutPut.setText("EIH Calibrate Finished\nT:\n"+str(self.Camera2Gripper))
    #     np.save(sys.path[0]+'/Camera2Gripper.npy',np.asarray(self.Camera2Gripper))
    #     fr.close()
        
    def SetInitPosF(self):
        try:
            self.Start2Stop()  #暫停取像
            self._arm_thread.goHome()
            self.Start2Stop()  
            # fr = open(sys.path[0]+"/calibration/PosSet.txt",'r+') 
            # dic = eval(fr.read())
            # fr.close()
            # #------Use this two position for safe movement
            # # self._arm_thread.cntrl.move_robot_pos(200775,-142192,166042,-1709969,-1927,435409, 500)
            # # self._arm_thread.cntrl.wait_move_end()
            # # self._arm_thread.cntrl.move_robot_pos(153906,191943,166042,-1709969,-1927,1301238, 500)
            # # self._arm_thread.cntrl.wait_move_end()
            # #----------------------------------------------
            # ObjectPos = StrToNdArray.StrToArray(dic['InitPos'])
            # print("Go Home: "+str(ObjectPos))
            # self._arm_thread.cntrl.move_robot_pos(ObjectPos[0],ObjectPos[1],ObjectPos[2],ObjectPos[3],ObjectPos[4],ObjectPos[5],1000)
            # self._arm_thread.cntrl.wait_move_end()
            # self._arm_thread.goHome()
        except Exception as e:
            print(str(e))

    def GetPosF(self):
        try:
            self.OutPut.setText("pos="+str(self._arm_thread.testArm()))
        except Exception as e:
            print(str(e))

    def ConnectRobotArmF(self):
        try:
            if(self.ConnectRobtoArm.text() =="Connect TM_Arm"):
                self.Start2Stop()  #暫停取像
                self._arm_thread = ArmThread()
                self._arm_thread.grip.gripper_reset()
                self._arm_thread.start()
                self._arm_thread.testArm()
                # self._arm_thread.goHome()
                self._arm_thread.release_trigger.connect(self.updateRGBDFrame)
                self.ConnectRobtoArm.setText("Disconnect TM_Arm")
                self.Start2Stop() #恢復取像
            else:
                self._arm_thread.robot.pasue()
                self._arm_thread.grip.gripper_off()
                self.ConnectRobtoArm.setText("Connect TM_Arm")
        except Exception as e:
            print(str(e))

    def Start2Stop(self):
            if(self.CameraOn.text() == "Camera ON"):
                self.CameraOn.setText("Camera OFF")
                print("Camera connected.")
                if(self._stream_thread._running == True):
                    self._stream_thread.start()
                else:
                    self._stream_thread.restart()
            else:
                self.CameraOn.setText("Camera ON")
                print("Camera disconnected.")
                self._stream_thread.pause()

    def updateRGBDFrame(self, rgbd_image):
        QApplication.processEvents()
        self._rgb_image = QImage(rgbd_image[0][:], rgbd_image[0].shape[1], rgbd_image[0].shape[0], QImage.Format_RGB888)
        #self._d_image = QImage(rgbd_image[2][:], rgbd_image[2].shape[1], rgbd_image[2].shape[0], rgbd_image[2].shape[1]*1, QImage.Format_RGB888)
        self._d_image = QImage(rgbd_image[1][:], rgbd_image[1].shape[1], rgbd_image[1].shape[0], QImage.Format_Grayscale8)
        #qimage = QImage(depth_map_normalized.data, width, height, QImage.Format_Grayscale8)
        #bytesPerline = channel * widthself.base_frame_id = "panda_link0" #用途不明
        self.tool0_frame_id = "panda_link8" #用途不明

        
        #self.finger_depth = rospy.geT_tool0_tcpt_param("~finger_depth") 這是從.yaml的設定檔案拉參數出來用的方式
        self.finger_depth = 0.04 # 單位m

        #self.size = 6.0 * self.finger_depth
        self.size = 0.3 #workspace空間


        #self.robot = RobotControl_func.RobotControl_Func() #手臂控制
        ### TODO tc-huang
        #self.grip = Gripper()
        ###

        #self.setup_panda_control()
        self.tf_tree = ros_utils.TransformTree()
        self.tsdf_server = TSDFServer()
        
        try:
            #TODO: one point is too small, 7*7 all red
            
            if mode == 0: # Draw a point
                for i in range(-3, 3, 1):
                    for j in range(-3, 3, 1):
                        self._rgb_image.setPixel(realX + i,realY + j, qRgb(255, 0, 0))  #在原始未縮放的圖片上畫圖
                        #self._d_image.setPixel(realX + i, realY + j, qRgb(255, 0, 0))
            # if mode == 1: # Draw a BBox  但應該用不到
            #     if realX > releaseX:
            #         x1 = releaseX
            #         x2 = realX
            #     else:
            #         x2 = releaseX
            #         x1 = realX
            #     if realY > releaseY:
            #         y1 = releaseY
            #         y2 = realY
            #     else:
            #         y2 = releaseY
            #         y1 = realY
            #     for i in range(-2, 2, 1):
            #         for j in range(-2, 2, 1):
            #             for w in range(0, x2 - x1, 1):
            #                 self._rgb_image.setPixel(x1 + w + i, y1 + j, qRgb(255, 0, 0))
            #                 self._rgb_image.setPixel(x1 + w + i, y2 + j, qRgb(255, 0, 0))
            #                 self._d_image.setPixel(x1 + w + i, y1 + j, qRgb(255, 0, 0))
            #                 self._d_image.setPixel(x1 + w + i, y2 + j, qRgb(255, 0, 0))
            #             for h in range(0, y2 - y1, 1):
            #                 self._rgb_image.setPixel(x1 + i, y1 + h + j, qRgb(255, 0, 0))
            #                 self._rgb_image.setPixel(x2 + i, y1 + h + j, qRgb(255, 0, 0))
            #                 self._d_image.setPixel(x1 + i, y1 + h + j, qRgb(255, 0, 0))
            #                 self._d_image.setPixel(x2 + i, y1 + h + j, qRgb(255, 0, 0))
        except:
            pass
        
        self.RGBFrame.setPixmap(QPixmap.fromImage(self._rgb_image).scaled(890,500))
        self.DepthFrame.setPixmap(QPixmap.fromImage(self._d_image).scaled(409,230))
        # self.RGBFrame.setPixmap(QPixmap.fromImage(self._rgb_image).scaled(890,500,Qt.KeepAspectRatio))
        # self.DepthFrame.setPixmap(QPixmap.fromImage(self._d_image).scaled(409,230,Qt.KeepAspectRatio))
        

    def mouseClicked(self,event):  
        global realX, realY, mode,imgX,imgY
        mode = 0
        # realX = int(event.pos().x() / 889 * 640 )  #889,500是視窗大小
        # realY = int(event.pos().y() / 500 * 480 )
        realX = int(event.pos().x() * 640/890 )  #890,500是視窗大小
        realY = int(event.pos().y() * 480/500 )
        imgX = event.pos().x()
        imgY = event.pos().y()
        print("-------Click-------\n",event.pos().x())
        print(event.pos().y())
        print("-------real-------\n",realX)
        print(realY)
        mode = 0
        self.getDepthF()

    # def mouseRelease(self,event):
    #     global releaseX , releaseY, mode
    #     releaseX = int(event.pos().x() * 640/889 )
    #     releaseY = int(event.pos().y() * 480/500 )
    #     if releaseX == realX and releaseY == realY:
    #         mode = 0
    #         self.getDepthF()
    #         # A point for estimateCoordinate
    #     else:
    #         mode = 1
    #         # A BBox for depthCalculate
    #         self.getDepthF()
        # print("-------Release-------\n",event.pos().x())
        # print(event.pos().y())

    def PenTouchTestF(self): #下爪點位測試
        intrinsic=np.load(sys.path[0]+'/INS.npy')
        Camera2Gripper=np.load(sys.path[0]+'/Camera2Gripper.npy')
        [x,y,z,u,v,w]=self._arm_thread.robot.get_TMPos()
        Estimate_Coord= EIHCali.EstimateCoord(realX,realY,intrinsic,float(self.depth_value),Camera2Gripper,Current_pos=[x,y,z,u,v,w])   
        Estimate_Coord=np.asarray(Estimate_Coord)
        print(Estimate_Coord)
        print(type(Estimate_Coord))
        self.OutPut.setText("Estimate_Coord:\n"+str(Estimate_Coord))
        print(Estimate_Coord[0][0])
        print(Estimate_Coord[1][0])
        self._arm_thread.robot.set_TMPos([Estimate_Coord[0][0],Estimate_Coord[1][0],40,u,v,w])
        # try:
            
        #     # self._arm_thread.cntrl.move_robot_pos(str(int(EstimateCoordResult[0])), str(int(EstimateCoordResult[1])), str(int(EstimateCoordResult[2])), -1710090 ,-24420 ,1620220 , 500)
        #     # self._arm_thread.cntrl.wait_move_end()
        # except Exception as e:
        #     print(str(e))

    def convertQImageToMat(self,incomingImage):
        incomingImage = incomingImage.convertToFormat(4)
        width = incomingImage.width()
        height = incomingImage.height()
        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height,width,4)  # Copies the data
        return arr

    def CameraCalibrationF(self):
        self.Start2Stop() #用這個去控制thread
        #self.CameraCalibration.setText("20") #调试时直接用拍摄好的照片做校正
   
        self.intrinsic,self.distCoeffs=EIHCali.CameraCalibration(sys.path[0])
        #to do np.save()
        
        np.save(sys.path[0]+'/INS.npy',np.asarray(self.intrinsic))
        np.save(sys.path[0]+'/dist.npy',np.asarray(self.distCoeffs))
        dict = {'intrinsic': str(self.intrinsic), 'distCoeffs': str(self.distCoeffs)}
        fw = open(sys.path[0]+"/CameraCali.txt",'w+')
        fw.write(str(dict))
        fw.close()
            #fr = open(sys.path[0]+"/calibration/CameraCali.txt",'r+')
            #dic = eval(fr.read())
            #print(dic['MTX'])
            #fr.close()
        self.OutPut.append(dict['intrinsic'])
        self.OutPut.setText("Camera Calibrate Finished")
        cv2.destroyAllWindows() 
        self.Start2Stop()

    def SaveImagesF(self):
        try:
            if not os.path.exists(sys.path[0]+'/Saved_IMG'):
                os.makedirs(sys.path[0]+'/Saved_IMG')
            imageInFrame = self._rgb_image
            DepthImageInFrame = self._d_image
            image = self.convertQImageToMat(imageInFrame)
            depth_image = self.convertQImageToMat(DepthImageInFrame)
            #ticks = time.time()
            # cv2.imwrite(sys.path[0]+"/saved/RGB" + str(ticks) + ".jpg",image)
            # cv2.imwrite(sys.path[0]+"/saved/D" + str(ticks) + ".jpg",depth_image)
            cv2.imwrite(sys.path[0]+"/Saved_IMG/" + str(self.cali_img_id) + ".png",image)
            #cv2.imwrite(sys.path[0]+"/Saved_IMG/D" + str(self.cali_img_id) + ".png",depth_image) #深度圖使用
            self.OutPut.setText("Saved:"+str(self.cali_img_id))
            self.cali_img_id +=1
        except Exception as e:
            self.OutPut.setText(str(e))

    def GripperOpenF(self):
        try:
            self._stream_thread.pause() #暫停取像
            print("Open Gripper")
            self._arm_thread.grip.gripper_on()
            self._stream_thread.restart() #恢復取像
        except Exception as e:
            self.OutPut.setText(str(e))

    def GripperCloseF(self):
        try:
            self._stream_thread.pause() #暫停取像
            print("Close Gripper")
            self._arm_thread.grip.gripper_off()
            self._stream_thread.restart() #恢復取像
        except Exception as e:
            self.OutPut.setText(str(e))




class PandaGraspController(object):
    def __init__(self, args):

        self.base_frame_id = "panda_link0" #用途不明
        self.tool0_frame_id = "panda_link8" #用途不明

        
        #self.finger_depth = rospy.geT_tool0_tcpt_param("~finger_depth") 這是從.yaml的設定檔案拉參數出來用的方式
        self.finger_depth = 0.04 # 單位m

        #self.size = 6.0 * self.finger_depth
        self.size = 0.3 #workspace空間


        #self.robot = RobotControl_func.RobotControl_Func() #手臂控制
        ### TODO tc-huang
        #self.grip = Gripper()
        ###

        #self.setup_panda_control()
        self.tf_tree = ros_utils.TransformTree()
        self.tsdf_server = TSDFServer()
        self.plan_grasps = VGN(args.model, rviz=True)
        
        rospy.loginfo("Ready to take action")

    def run(self, robot_arm,T_cam2gripper, gripper):
        vis.clear()
        vis.draw_workspace(self.size)
        #self.pc.move_gripper(0.08)
        #self.pc.home()
        #沒墊東西的位置
        #self.robot.set_TMPos([289.77935791015625, -464.6210632324219, 296.10107421875, 3.0886695174594117, 0.045914965304000084, 0.7913568531784461])
        # TODO: set pose here
        # self.robot.set_TMPos([269.2006225585938, -464.2568359375, 422.3720703125, 3.10223965479391, 0.023221647426189884, 0.8183750884904907])
        ### TODO: tc-huang
        robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
        logging.info("[Robot] Move to initial pose by joint configure")
        ###

        tsdf, pc = self.acquire_tsdf()
        vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(np.asarray(pc.points))
        rospy.loginfo("Reconstructed scene")

        state = State(tsdf, pc) #模擬環境跑一次看看
        grasps, scores, planning_time = self.plan_grasps(state)
        vis.draw_grasps(grasps, scores, self.finger_depth)
        rospy.loginfo("Planned grasps")

        if len(grasps) == 0:
            rospy.loginfo("No grasps detected")
            return

        grasp, score = self.select_grasp(grasps, scores)
        vis.draw_grasp(grasp, score, self.finger_depth)
        rospy.loginfo("Selected grasp")
        ### TODO tc-huang
        #self.grip.gripper_on()
        ###
        # self.pc.home()
        label = self.execute_grasp(grasp, robot_arm=robot_arm, Camera2Gripper=T_cam2gripper, gripper=gripper)
        rospy.loginfo("Grasp execution")
        rospy.sleep(1)
        # if self.robot_error:
        #     self.recover_robot()
        #     return

        # if label:
        #     self.drop()
        # self.pc.home()
        #self.robot.set_TMPos([289.77935791015625, -464.6210632324219, 296.10107421875, 3.0886695174594117, 0.045914965304000084, 0.7913568531784461]) #home
        ### TODO tc-huang
        robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
        logging.info("[Robot] Move to initial pose by joint configure")
        ###serial

    def acquire_tsdf(self): #移動去其他視角拍照
        fr = open("./scripts/PosSet.txt", 'r+')  #這時還沒有抓取joint的function所以使用position當作點位
        
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
        rospy.sleep(2.0)
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

        # vis.draw_quality(qual_voserial)
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

    def execute_grasp(self, grasp, robot_arm, Camera2Gripper, gripper):  #夾取用的 grasp是從world座標

        T_task_grasp = grasp.pose
        print("模型測出來的位置")
        print(T_task_grasp.translation)
        #T_base_grasp = self.T_base_task
        
        print("grasp tsdf位置",grasp)
        Base_point_t,Base_R_degree=self.EstimateCoord(T_task_grasp, Camera2Gripper, robot_arm)
        print("Base_point_t",Base_point_t)
        print("Base_R_degree",Base_R_degree)
        #T_base_grasp = self.T_base_task *T_task_grasp
        
        #T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])  #移動到前方
        #T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])   #夾起來後往後

        # T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        # T_base_retreat = T_base_grasp * T_grasp_retreat
        # point=T_base_pregrasp * T_tcp_tool0
        # print(point.translation)
        #移動到指定位置
        #或是直接使用T_base_grasp
        # point=T_base_pregrasp * T_tcp_tool0
        
        #degree=T_base_grasp.rotation.as_euler('xyz', degrees=False)

        # t_task_grasp = T_task_grasp.translation
        # R_task_grasp = T_task_grasp.rotation
        # modifed_Rz_degree = Base_R_degree[2]
        # print("original",Base_R_degree)

        # while modifed_Rz_degree<-90 :
        #     modifed_Rz_degree+=180
        #     print("-90")
        # while modifed_Rz_degree>90:
        #     modifed_Rz_degree-=180
        #     print("+90")

        # print("modifed_Rz_degree",modifed_Rz_degree)
        ### TODO: tc-huang
        robot_arm.move_to_pose(
            Tx=Base_point_t[0],
            Ty=Base_point_t[1],
            Tz = Base_point_t[2],
            Rx=Base_R_degree[0],
            Ry=Base_R_degree[1],
            # Rz=modifed_Rz_degree,
            Rz=Base_R_degree[2],

            speed=SPEED)
        #TODO: gripper
        success = gripper.close()
        time.sleep(3)
        robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)  #初始位置

        robot_arm.move_to_joint_config(Place_JOINT_CONFIGURE, SPEED)  #擺放位置
        success = gripper.on()
        # robot_arm.move_to_joint_config([-126330, -19968, 12272, -1538, -100260, 46100 + 52680], SPEED)
        
        ###

    def approach_grasp(self, T_base_grasp):
        self.pc.goto_pose(T_base_grasp * self.T_tcp_tool0)

    def drop(self):  #放置位置
        self.pc.goto_joints(
            [0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931], 0.2, 0.2
        )
        self.pc.move_gripper(0.08)

    def EstimateCoord(self,T_task_grasp, Camera2Gripper, robot_arm):  #計算座標
        # TODO tc-huang
        tcp_pose = robot_arm.controller.get_robot_pose(tool_num=0)
        print(f"TCP POSE: {tcp_pose}")
        tvec = tcp_pose[:3]

        x_angle, y_angle, z_angle = tcp_pose[3:]
        rvec = Rotation.from_euler(
            "xyz", [x_angle, y_angle, z_angle], degrees=True
        ).as_rotvec(degrees=False)
        
        Gripper2Base= np.eye(4)
        Gripper2Base[:3, :3] = Rotation.from_euler(
            "xyz", [x_angle, y_angle, z_angle], degrees=True
        ).as_matrix()
        Gripper2Base[:3, 3] =np.array(tvec).T

        print(f"TCP POSE mat: {Gripper2Base}")

        # Task2Camera = np.array( #改成算出來的
        #     [
        #         [1, 0, 0, 0],
        #         [0, 1, 0, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1]
        #     ]
        # )
        print("---------")
        print(T_cam_task_m.as_matrix())
        Task2Camera_r = T_cam_task_m.as_matrix()[:3,:3]
        Task2Camera_t = T_cam_task_m.as_matrix()[:3,3]  #公尺
        print(Task2Camera_r)
        print(Task2Camera_t*1000)

        

        Task2Camera=np.r_[np.c_[Task2Camera_r, Task2Camera_t*1000], [[0, 0, 0, 1]]]
        # grasppoint = np.array([T_task_grasp.translation[0]*1000,T_task_grasp.translation[1]*1000, T_task_grasp.translation[2]*1000, 1]).T
        # grasppoint = np.array([0,0,0,1]).T

        #TODO 限制z軸旋轉
        Grasp2task = np.eye(4)
        T_task_grasp_t = T_task_grasp.as_matrix()[:3,3]
        T_task_grasp_r = T_task_grasp.as_matrix()[:3,:3]
        #TODO 未知
        T_task_grasp_Rx_deg, T_task_grasp_Ry_deg, T_task_grasp_Rz_deg = Rotation.from_matrix(T_task_grasp_r).as_euler('xyz', degrees=True)
        print("原始角度",T_task_grasp_Rx_deg, T_task_grasp_Ry_deg, T_task_grasp_Rz_deg )
        # while T_task_grasp_Rz_deg < 0:
        #     T_task_grasp_Rz_deg += 180
        #     print("-90")
        # while T_task_grasp_Rz_deg > 180:
        #     T_task_grasp_Rz_deg -= 180
        #     print("+90")
        # T_task_grasp_r = Rotation.from_euler('xyz', [T_task_grasp_Rx_deg, T_task_grasp_Ry_deg, T_task_grasp_Rz_deg], degrees=True).as_matrix()
        
        Grasp2task[:3,:3] = T_task_grasp_r
        Grasp2task[:3,3] = T_task_grasp_t * 1000

        # Grasp2task=np.r_[np.c_[T_task_grasp_r, T_task_grasp_t], [[0, 0, 0, 1]]]
        print("grasp2task",Grasp2task)

        tm = TransformManager()
        tm.add_transform("grasp", "task", Grasp2task)
        # tm.add_transform("gripper", "robot", Gripper2Base)
        # tm.add_transform("camera", "gripper", Camera2Gripper)
        # tm.add_transform("task", "camera", Task2Camera)

        #ee2object = tm.get_transform("end-effector", "object")

        ax = tm.plot_frames_in("task", s=100)
        ax.set_xlim((-1000, 1000))
        ax.set_ylim((-1000, 1000))
        ax.set_zlim((-1000, 1000))
        plt.show()


        Base_point_T = Gripper2Base @ Camera2Gripper @ Task2Camera @ Grasp2task @ offset    #從右邊往左看,相機座標到夾爪座標再到base座標
        print("Camera2Gripper", Camera2Gripper)
        print("Gripper2Base",Gripper2Base)
        print("Task2Camera",Task2Camera)
        Base_point_t = Base_point_T[:3, 3] #3x1 T
        Base_point_r = Base_point_T[:3,:3] #3x3
        Base_R_degree= Rotation.from_matrix(Base_point_r).as_euler('xyz',degrees=True)
        

        return Base_point_t,Base_R_degree


class TSDFServer(object):
    def __init__(self):
        self.cam_frame_id = "camera_depth_optical_frame"
        self.cam_topic_name = "/camera/depth/image_rect_raw"
        ### TODO change intrinsic tc-huang
        self.intrinsic = CameraIntrinsic(640, 480, 606.77737126, 606.70030146, 321.63287183, 236.95293136)
        self.distortion=np.array([ 4.90913179e-02 , 5.22699002e-01, -2.65209452e-03  ,1.13033224e-03,-2.17133474e+00])

        #

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
        #T_cam_task =  ArUco_detection.Img_ArUco_detect(img,self.intrinsic,self.distortion)
        #用來廣播目標平面到相機的關係，也就是相機外參
        self.tf_tree.broadcast_static(
            T_cam_task_m, self.cam_frame_id, "task"
        )


        self.low_res_tsdf.integrate(img, self.intrinsic, T_cam_task_m)
        self.high_res_tsdf.integrate(img, self.intrinsic, T_cam_task_m)


def main(args):
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController(args)

    ### TODO: tc-huang
    with open(str(CALIBRATION_PARAMS_SAVE_DIR / 'calibration_params.json')) as f:
        calibration_params = json.load(f)
        T_cam2gripper = np.array(calibration_params["T_cam2gripper"])
        T_target2cam = np.array(calibration_params["T_target2cam"])
        T_gripper2base = np.array(calibration_params["T_gripper2base"])
        intrinsic_matrix = np.array(calibration_params["intrinsic_matrix"])
        distortion_coefficients = np.array(calibration_params["distortion_coefficients"])
    logging.info("Init")
    gripper = Gripper2F85()
    logging.info("Connect")
    success = gripper.connect()
    logging.info("Reset")
    success = gripper.reset()

    robot_arm = MH5L(host=HOST, port=PORT)
    logging.info("[Robot] Init")
    robot_arm.power_on()
    logging.info("[Robot] Power on")
    robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
    logging.info("[Robot] Move to initial pose by joint configure")
    
    ###
    ### TODO: times
    for i in range(3):
        robot_arm.move_to_joint_config([-126330, -19968, 12272, -1538, -100260, 46100 + 52680], SPEED)
        panda_grasp.run(robot_arm, T_cam2gripper, gripper) 
    
    ### TODO tc-huang
    robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
    logging.info("[Robot] Move to initial pose by joint configure")

    robot_arm.power_off() 
    logging.info("[Robot] Power off")
    ### 
    


def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        S = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / S
        x = (R[2, 1] - R[1, 2]) * S
        y = (R[0, 2] - R[2, 0]) * S
        z = (R[1, 0] - R[0, 1]) * S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return np.array([w, x, y, z])
        

if __name__ == "__main__":
    rospy.init_node("rivz",anonymous=True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
