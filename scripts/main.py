import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from ui.mainwindow import Ui_MainWindow
import pyrealsense2 as rs
import numpy as np
import cv2
# from robot.controller import Controller
# from robot.gripper import Gripper
import time
import glob
import StrToArray
import calibration.EIHCali as EIHCali
import rospy
import RobotControl_func_ros1 as RobotControl_func
from gripper import Gripper

class RGBDThread(QThread):
    rgbd_trigger = pyqtSignal(object)
    def __init__(self):
        super(RGBDThread, self).__init__()
        self._mutex = QMutex()
        self._running = True
        self.pipeline = rs.pipeline()
        self.colorizer = rs.colorizer()
        self.config = rs.config()
        #self.config.enable_device('f0265339')
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        profile = self.pipeline.start(self.config)
        depth_sensor = profile.get_device().first_depth_sensor()
        # if depth_sensor.supports(rs.option.depth_units):
        #     depth_sensor.set_option(rs.option.depth_units, 0.0001)
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth units setted: ", '%.5f'%self.depth_scale)
        self.dmin = 1000.0
        self.dmax = 2400.0
        #--------For another Camera-----
        # self.dmin = 5000.0
        # self.dmax = 5200.0
        #-------------------------------
        self.scale = 1
    def __del__(self):
        self.pipeline.stop()
        self.wait()
    def run(self):
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
                    # depth_image_nobg = np.where(self.depth_image > self.clipping_distance, 0, self.depth_image)
                    # print(np.amax(depth_image_nobg))
                    # depth_image_nobg = np.uint8(depth_image_nobg)
                    # self.depth_image_3c = cv2.convertScaleAbs(np.dstack((depth_image_nobg,depth_image_nobg,depth_image_nobg)), alpha=0.03)

                    depth_image_nobg = np.where(self.depth_image == 0.0, self.dmax, self.depth_image)
                    depth_image_nobg = np.where(depth_image_nobg > self.dmax, self.dmax, depth_image_nobg)
                    depth_image_nobg = np.where(depth_image_nobg < self.dmin, self.dmin, depth_image_nobg)
                    depth_image_nobg = depth_image_nobg / (self.scale * 10)
                    depth_image_nobg = self.normalize(depth_image_nobg)

                    depth_image_3c = np.dstack((depth_image_nobg,depth_image_nobg,depth_image_nobg))
                    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                    # self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_3c, alpha=0.03), cv2.COLORMAP_JET)
                    # TODO: normalize to 0-255
                    self.color_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
                    depth_image_3c = cv2.cvtColor(depth_image_3c,cv2.COLOR_BGR2RGB)

                    self.rgbd_pair = [self.color_image,depth_image_3c,self.depth_image]
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
            if realX > releaseX:
                x1 = releaseX
                x2 = realX
            else:
                x2 = releaseX
                x1 = realX
            if realY > releaseY:
                y1 = releaseY
                y2 = realY
            else:
                y2 = releaseY
                y1 = realY
            scaleX1 = int((1.3 * x1 + 0.7 * x2) / 2 + 0.5)
            scaleX2 = int((0.7 * x1 + 1.3 * x2) / 2 + 0.5)
            scaleY1 = int((1.3 * y1 + 0.7 * y2) / 2 + 0.5)
            scaleY2 = int((0.7 * y1 + 1.3 * y2) / 2 + 0.5)
            # print(scaleX1, scaleX2, scaleY1, scaleY2)
            frames = self.pipeline.wait_for_frames()
            New_depth_image = np.asanyarray(self.align.process(frames).get_depth_frame().get_data())
            # print(np.shape(New_depth_image))
            # plt.imshow(New_depth_image)
            # plt.show()
            depthmap = New_depth_image[scaleY1:scaleY2, scaleX1:scaleX2].astype(float) * self.depth_scale * 100
            depthmap = depthmap.flatten()
            # print("-----------BBox有效点-------------\n", np.shape(depthmap))
            # TODO: 去除异常值 0
            indices = np.where(depthmap == 0.0)
            depthmap = np.delete(depthmap, indices)
            # print("-----------去除0后大小-------------\n",np.shape(depthmap))
            # TODO: 去除异常值 保留25%~75%之间的数字
            Outliers1 = np.percentile(depthmap, 25)
            Outliers2 = np.percentile(depthmap, 75)
            # print(Outliers1,Outliers2)
            indices = np.where(depthmap > Outliers2)
            depthmap = np.delete(depthmap, indices)
            indices = np.where(depthmap < Outliers1)
            depthmap = np.delete(depthmap, indices)
            # print("-----------删除异常值后-------------\n",np.shape(depthmap))
            dist,_,_,_ = cv2.mean(depthmap)
            return "Depth: "+str('%.2f'%dist)+"cm"
        except Exception as e:
            return e
    def get_depth_profile(self):
        return self.depth_scale, self.d_intrin
    def normalize(self, depth):
        min_val = 30
        max_val = 255
        d_med = (np.max(depth) + np.min(depth)) / 2.0
        d_diff = depth - d_med
        depth_rev = np.copy(depth)
        depth_rev = d_diff * (-1) + d_med
        depth_rev = depth_rev - np.min(depth_rev)

        depth_rev = cv2.normalize(depth_rev, None, min_val, max_val, cv2.NORM_MINMAX, cv2.CV_8UC1)
        return depth_rev
        #return cv2.normalize(depth, None, min_val, max_val, cv2.NORM_MINMAX, cv2.CV_8UC1)

class ArmThread(QThread):
    # control signal: working signal of robotarm
    # Input: position of bounding box
    # output: finish signal
    release_trigger = pyqtSignal(object)

    def __init__(self):
        super(ArmThread,self).__init__()
        self._mutex = QMutex()
        self.cntrl = Controller()
        self.grip = Gripper()
        self.trans_mat = np.array([
            [0.6219,-0.0021,0.],
            [-0.0028,-0.6218,0.],
            [-337.3547,-163.6015,1.0]
        ])
        self.baseline_depth = 5850
        self.pose = None
        self._running = True
        self.release = False

    def initArmGripper(self):
        # to init arm position
        # init gripper
        self.grip.gripper_reset()
        self.cntrl.power_on()

    def run(self):
        while self.running():
            if self.pose == None:
                # print("don't move")
                continue
            else:
                self.pick(self.pose)
                self.pose = None

    def testArm(self):
        rcv = self.cntrl.get_robot_pos()
        pos = self.cntrl.robot_info
        print("pos=",pos)
        return rcv

    def calPosXY(self,camera_point):
        # print(camera_point)
        camera_xy = [camera_point[0],camera_point[1]]
        camera_xy.append (1.0)
        # print(camera_point)
        arm_point = np.array (np.array(camera_xy)) @ self.trans_mat
        print ("arm_point",arm_point)
        return arm_point

    def pick(self,camera_point):
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

    def goHome(self):
        self.cntrl.move_robot_pos ('2883','-246016','166040','-1709973','-1929','-104740',500)
        self.cntrl.wait_move_end ()

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
        self._stream_thread = RGBDThread()
        self._stream_thread.rgbd_trigger.connect(self.updateRGBDFrame)
        self.RGBFrame.mousePressEvent = self.mouseClicked
        self.RGBFrame.mouseReleaseEvent = self.mouseRelease
        #--------------------Bind UI with func（Start）--------------------------------
        self.CameraOn.clicked.connect(self.Start2Stop) #连接相机
        self.CameraCalibration.clicked.connect(self.CameraCalibrationF) #Camera Calibration
        self.SaveImages.clicked.connect(self.SaveImagesF)  # Save Image
        self.GripperOpen.clicked.connect(self.GripperOpenF) #Open Gripper
        self.GripperClose.clicked.connect(self.GripperCloseF)  # Close Gripper
        self.ConnectRobtoArm.clicked.connect(self.ConnectRobotArmF) #connect RobotArm
        self.SetInitPos.clicked.connect(self.SetInitPosF) #Set Init Position
        self.GetPos.clicked.connect(self.GetPosF) #Get Current Position
        self.AutoETHCalib.clicked.connect(self.AutoETHCalibF)
        self.DebugCalib.clicked.connect(self.DebugCalibF)
        self.TestCalibration.clicked.connect(self.TestCalibrationF)
        self.FindPlane.clicked.connect(self.FindPlaneF)
        self.PenTouchTest.clicked.connect(self.PenTouchTestF)
        self.DepthTest.clicked.connect(self.getDepthF)
        # --------------------Bind UI with func（End）--------------------------------
    def getDepthF(self):
        self.OutPut.setText(str(self._stream_thread.getDepth()))
    def FindPlaneF(self):
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

    def TestCalibrationF(self):
        #TODO: Gripper grasp Plane
        fr = open(sys.path[0] + "/calibration/PosSet.txt", 'r+')
        dic = eval(fr.read())
        fr.close()
        GetPlanePos = StrToArray.StrToArray(dic['GetPlanePos'])
        ObjectPos = StrToArray.StrToArray(dic['CaliPos'])
        self._arm_thread.cntrl.move_robot_pos(GetPlanePos[0], GetPlanePos[1], GetPlanePos[2], GetPlanePos[3],
                                              GetPlanePos[4], GetPlanePos[5], 2000)
        self._arm_thread.cntrl.wait_move_end()
        self.GripperCloseF()
        time.sleep(1)  # time for Gripper Close
        #TODO: Make 3 move
        self._arm_thread.cntrl.move_robot_pos(ObjectPos[0][0], ObjectPos[0][1], ObjectPos[0][2], ObjectPos[0][3], ObjectPos[0][4], ObjectPos[0][5], 500)
        self._arm_thread.cntrl.wait_move_end()
        imageInFrame = self._rgb_image
        image = self.convertQImageToMat(imageInFrame)
        cv2.imshow("ETH Calibration Test",image)
        cv2.waitKey(50)
        #TODO: 不知道什么问题，但是要用waitkey刷新一下RGB Frame
        imageInFrame = self._rgb_image
        image = self.convertQImageToMat(imageInFrame)
        cv2.imshow("ETH Calibration Test", image)
        cv2.imwrite(sys.path[0] + "/calibration/EyeToHandCali/Test1.jpg", image)
        cv2.waitKey(50)

        self._arm_thread.cntrl.move_robot_pos(ObjectPos[5][0], ObjectPos[5][1], ObjectPos[5][2], ObjectPos[5][3],
                                              ObjectPos[5][4], ObjectPos[5][5], 500)
        self._arm_thread.cntrl.wait_move_end()
        imageInFrame = self._rgb_image
        image = self.convertQImageToMat(imageInFrame)
        cv2.imshow("ETH Calibration Test", image)
        cv2.waitKey(50)
        #TODO: 不知道什么问题，但是要用waitkey刷新一下RGB Frame
        imageInFrame = self._rgb_image
        image = self.convertQImageToMat(imageInFrame)
        cv2.imshow("ETH Calibration Test", image)
        cv2.imwrite(sys.path[0] + "/calibration/EyeToHandCali/Test2.jpg", image)
        cv2.waitKey(50)

        self._arm_thread.cntrl.move_robot_pos(ObjectPos[8][0], ObjectPos[8][1], ObjectPos[8][2], ObjectPos[8][3],
                                              ObjectPos[8][4], ObjectPos[8][5], 500)
        self._arm_thread.cntrl.wait_move_end()
        imageInFrame = self._rgb_image
        image = self.convertQImageToMat(imageInFrame)
        cv2.imshow("ETH Calibration Test", image)
        cv2.waitKey(50)
        #TODO: 不知道什么问题，但是要用waitkey刷新一下RGB Frame
        imageInFrame = self._rgb_image
        image = self.convertQImageToMat(imageInFrame)
        cv2.imshow("ETH Calibration Test", image)
        cv2.imwrite(sys.path[0] + "/calibration/EyeToHandCali/Test3.jpg", image)
        cv2.destroyAllWindows()
        #TODO:Calculate the result
        ETHCali.TestETHCali()
        #TODO: Put Chessboard Back
        self._arm_thread.cntrl.move_robot_pos(GetPlanePos[0], GetPlanePos[1], GetPlanePos[2], GetPlanePos[3],
                                              GetPlanePos[4], GetPlanePos[5], 2000)
        self._arm_thread.cntrl.wait_move_end()
        self.GripperOpenF()
        time.sleep(1)  # Time for gripper open
        self.SetInitPosF()

    def DebugCalibF(self):
        result = ETHCali.ETHCali()
        self.OutPut.setText("ETH Calibrate Finished\nT:\n" + str(result))
        ETHCali.TestETHCali()

    def AutoETHCalibF(self):
        try:
            fr = open(sys.path[0] + "/calibration/PosSet.txt", 'r+')
            dic = eval(fr.read())
            fr.close()
            ObjectPos = StrToArray.StrToArray(dic['CaliPos'])
            GetPlanePos = StrToArray.StrToArray(dic['GetPlanePos'])
            self._arm_thread.cntrl.move_robot_pos(GetPlanePos[0], GetPlanePos[1], GetPlanePos[2], GetPlanePos[3], GetPlanePos[4], GetPlanePos[5], 2000)
            self._arm_thread.cntrl.wait_move_end()
            self.GripperCloseF()
            time.sleep(1) #time for Gripper Close
            i = 0
            j = len(ObjectPos)
            # print(j)
            for Pose in ObjectPos:
                self._arm_thread.cntrl.move_robot_pos(Pose[0], Pose[1], Pose[2], Pose[3], Pose[4], Pose[5], 500)
                self._arm_thread.cntrl.wait_move_end()
                print('Position:' + str(Pose))
                # print('I am in ' + str(self._arm_thread.testArm()))
                i += 1
                self.OutPut.setText("Begin ETH Calibration(" + str(i) + "/" + str(j) + ")")
                w = 11
                h = 8
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)
                objp = np.zeros((w * h, 3), np.float32)
                objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
                # 储存棋盘格角点的世界坐标和图像坐标对
                try:
                    imageInFrame = self._rgb_image
                    image = self.convertQImageToMat(imageInFrame)
                    cv2.imshow("ETH Calibration", image)
                    cv2.waitKey(50)
                    #在这里刷新一下Frame
                    imageInFrame = self._rgb_image
                    image = self.convertQImageToMat(imageInFrame)
                    cv2.imshow("ETH Calibration", image)
                    cv2.waitKey(50)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # 找到棋盘格角点
                    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
                    # 如果找到足够点对，将其存储起来
                    if ret == True:
                        cv2.imwrite(sys.path[0] + "/calibration/EyeToHandCali/" + str(i) + ".jpg",image)
                        print("saved:"+str(i)+"\n----------")
                        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        cv2.drawChessboardCorners(image, (w, h), corners, ret)
                        cv2.imshow("ETH Calibration", image)
                        cv2.waitKey(300)
                    else:
                        print("ERROR:No chessboard was found\n----------BREAK----------")
                except Exception as e:
                    self.OutPut.setText(str(e))
            cv2.destroyAllWindows()
            self._arm_thread.cntrl.move_robot_pos(GetPlanePos[0], GetPlanePos[1], GetPlanePos[2], GetPlanePos[3],GetPlanePos[4], GetPlanePos[5], 2000)
            self._arm_thread.cntrl.wait_move_end()
            self.GripperOpenF()
            time.sleep(1) #TODO: Time for gripper open
            self.SetInitPosF()
            result = ETHCali.ETHCali()
            self.OutPut.setText("ETH Calibrate Finished\nT:\n"+str(result))
        except Exception as e:
            print(str(e))

    def SetInitPosF(self):
        try:
            fr = open(sys.path[0]+"/calibration/PosSet.txt",'r+')
            dic = eval(fr.read())
            fr.close()
            #------Use this two position for safe movement
            # self._arm_thread.cntrl.move_robot_pos(200775,-142192,166042,-1709969,-1927,435409, 500)
            # self._arm_thread.cntrl.wait_move_end()
            # self._arm_thread.cntrl.move_robot_pos(153906,191943,166042,-1709969,-1927,1301238, 500)
            # self._arm_thread.cntrl.wait_move_end()
            #----------------------------------------------
            ObjectPos = StrToArray.StrToArray(dic['InitPos'])
            print("Go Home: "+str(ObjectPos))
            self._arm_thread.cntrl.move_robot_pos(ObjectPos[0],ObjectPos[1],ObjectPos[2],ObjectPos[3],ObjectPos[4],ObjectPos[5],1000)
            self._arm_thread.cntrl.wait_move_end()
            # self._arm_thread.goHome()
        except Exception as e:
            print(str(e))

    def GetPosF(self):
        try:
            self.OutPut.setText(str(self._arm_thread.testArm()))
        except Exception as e:
            print(str(e))

    def ConnectRobotArmF(self):
        try:
            if(self.ConnectRobtoArm.text() =="Connect Yaskawa"):
                self._arm_thread = ArmThread()
                self._arm_thread.initArmGripper()
                self._arm_thread.start()
                self._arm_thread.testArm()
                # self._arm_thread.goHome()
                self._arm_thread.release_trigger.connect(self.updateRGBDFrame)
                self.ConnectRobtoArm.setText("Disconnect Yaskawa")
            else:
                self._arm_thread.cntrl.power_off()
                self._arm_thread.grip.gripper_off()
                self.ConnectRobtoArm.setText("Connect Yaskawa")
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
        self._rgb_image = QImage(rgbd_image[0][:], rgbd_image[0].shape[1], rgbd_image[0].shape[0],rgbd_image[0].shape[1] * 3, QImage.Format_RGB888)
        self._d_image = QImage(rgbd_image[1][:], rgbd_image[1].shape[1], rgbd_image[1].shape[0], rgbd_image[1].shape[1] * 3, QImage.Format_RGB888)
        try:
            #TODO: one point is too small, 7*7 all red
            if mode == 0: # Draw a point
                for i in range(-3, 3, 1):
                    for j in range(-3, 3, 1):
                        self._rgb_image.setPixel(realX + i,realY + j, qRgb(255, 0, 0))
                        self._d_image.setPixel(realX + i, realY + j, qRgb(255, 0, 0))
            if mode == 1: # Draw a BBox
                if realX > releaseX:
                    x1 = releaseX
                    x2 = realX
                else:
                    x2 = releaseX
                    x1 = realX
                if realY > releaseY:
                    y1 = releaseY
                    y2 = realY
                else:
                    y2 = releaseY
                    y1 = realY
                for i in range(-2, 2, 1):
                    for j in range(-2, 2, 1):
                        for w in range(0, x2 - x1, 1):
                            self._rgb_image.setPixel(x1 + w + i, y1 + j, qRgb(255, 0, 0))
                            self._rgb_image.setPixel(x1 + w + i, y2 + j, qRgb(255, 0, 0))
                            self._d_image.setPixel(x1 + w + i, y1 + j, qRgb(255, 0, 0))
                            self._d_image.setPixel(x1 + w + i, y2 + j, qRgb(255, 0, 0))
                        for h in range(0, y2 - y1, 1):
                            self._rgb_image.setPixel(x1 + i, y1 + h + j, qRgb(255, 0, 0))
                            self._rgb_image.setPixel(x2 + i, y1 + h + j, qRgb(255, 0, 0))
                            self._d_image.setPixel(x1 + i, y1 + h + j, qRgb(255, 0, 0))
                            self._d_image.setPixel(x2 + i, y1 + h + j, qRgb(255, 0, 0))
        except:
            pass
        self.RGBFrame.setPixmap(QPixmap.fromImage(self._rgb_image).scaled(889,500,Qt.KeepAspectRatio))
        self.DepthFrame.setPixmap(QPixmap.fromImage(self._d_image).scaled(409,230,Qt.KeepAspectRatio))
        QApplication.processEvents()

    def mouseClicked(self,event):
        global realX, realY, mode
        mode = 0
        realX = int(event.pos().x() / 889 * 1280 + 0.5)
        realY = int(event.pos().y() / 500 * 720 + 0.5)
        # print("-------Click-------\n",event.pos().x())
        # print(event.pos().y())

    def mouseRelease(self,event):
        global releaseX , releaseY, mode
        releaseX = int(event.pos().x() / 889 * 1280 + 0.5)
        releaseY = int(event.pos().y() / 500 * 720 + 0.5)
        if releaseX == realX and releaseY == realY:
            mode = 0
            # A point for estimateCoordinate
        else:
            mode = 1
            # A BBox for depthCalculate
            self.getDepthF()
        # print("-------Release-------\n",event.pos().x())
        # print(event.pos().y())

    def PenTouchTestF(self):
        try:
            EstimateCoordResult = ETHCali.EstimateCoord(realX,realY)
            print(EstimateCoordResult)
            # self._arm_thread.cntrl.move_robot_pos(str(int(EstimateCoordResult[0])), str(int(EstimateCoordResult[1])), str(int(EstimateCoordResult[2])), -1710090 ,-24420 ,1620220 , 500)
            # self._arm_thread.cntrl.wait_move_end()
        except:
            print("No point select!")
    def convertQImageToMat(self,incomingImage):
        incomingImage = incomingImage.convertToFormat(4)
        width = incomingImage.width()
        height = incomingImage.height()
        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height,width,4)  # Copies the data
        return arr

    def CameraCalibrationF(self):
        self.Start2Stop()
        #self.CameraCalibration.setText("16") #调试时直接用拍摄好的照片做校正
        w= 11
        h = 8
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,0.001)
        objp = np.zeros((w * h,3),np.float32)
        objp[:,:2] = np.mgrid[0:w,0:h].T.reshape (-1,2)
        # 储存棋盘格角点的世界坐标和图像坐标对
        objpoints = []  # 在世界坐标系中的三维点
        imgpoints = []  # 在图像平面的二维点
        try:
            imageInFrame = self._rgb_image
            if (self.CameraCalibration.text () == "Camera Calib"):
                self.CameraCalibration.setText ("1")
                self.OutPut.setText ("Begin Camera Calibration(1/15)")
            image = self.convertQImageToMat(imageInFrame)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # 找到棋盘格角点
            ret,corners = cv2.findChessboardCorners(gray,(w,h),None)
            # 如果找到足够点对，将其存储起来
            if ret == True:
                cv2.imwrite(sys.path[0] + "/calibration/CameraCali/" + self.CameraCalibration.text() + ".jpg",image)
                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                objpoints.append(objp)
                imgpoints.append(corners)
                # 将角点在图像上显示
                cv2.drawChessboardCorners(image,(w,h),corners,ret)
                cv2.imshow("Camera Calibration",image)
                self.CameraCalibration.setText(str(int(self.CameraCalibration.text())+1))
                self.OutPut.setText ("Begin Camera Calibration("+self.CameraCalibration.text()+"/15)")
            else:
                print("No chessboard was found")
        except Exception as e:
            self.OutPut.setText(str(e))
        if (self.CameraCalibration.text() == "16"): #15张图片收集完成，开始做Camera Calibration把算出来的结果储存到/calibration/CameraCali/CameraCali.txt
            self.CameraCalibration.setText("Camera Calibration")
            self.OutPut.setText("Camera Calibrate Finished")
            #读取15张照片，做Camera Calibration
            images = glob.glob(sys.path[0] +'/calibration/CameraCali/*.jpg')
            for fname in images:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
                if ret == True:
                    cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    cv2.drawChessboardCorners(img,(w,h),corners,ret)
                    cv2.imshow("Camera Calibration",img)
                    cv2.waitKey(5)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            dict = {'MTX': str(mtx), 'Dist': str(dist)}
            fw = open(sys.path[0]+"/calibration/CameraCali.txt",'w+')
            fw.write(str(dict))
            fw.close()
            #fr = open(sys.path[0]+"/calibration/CameraCali.txt",'r+')
            #dic = eval(fr.read())
            #print(dic['MTX'])
            #fr.close()
            self.OutPut.append(dict['MTX'])
            cv2.destroyAllWindows()
        self.Start2Stop()

    def SaveImagesF(self):
        try:
            imageInFrame = self._rgb_image
            DepthImageInFrame = self._d_image
            image = self.convertQImageToMat(imageInFrame)
            depth_image = self.convertQImageToMat(DepthImageInFrame)
            ticks = time.time()
            cv2.imwrite(sys.path[0]+"/saved/RGB" + str(ticks) + ".jpg",image)
            cv2.imwrite(sys.path[0]+"/saved/D" + str(ticks) + ".jpg",depth_image)
            self.OutPut.setText("Saved:"+str(ticks))
        except Exception as e:
            self.OutPut.setText(str(e))

    def GripperOpenF(self):
        try:
            print("Open Gripper")
            self._arm_thread.grip.gripper_on()
        except Exception as e:
            self.OutPut.setText(str(e))

    def GripperCloseF(self):
        try:
            print("Close Gripper")
            self._arm_thread.grip.gripper_off()
        except Exception as e:
            self.OutPut.setText(str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
