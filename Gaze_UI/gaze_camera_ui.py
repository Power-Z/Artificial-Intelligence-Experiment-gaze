"""
说明：pyqt拍照应用测试代码
作者：张力
时间：2023-3-17
"""


# here put the import lib
import sys
from typing import Optional

import datetime
import logging
import pathlib

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from camera_gaze import Ui_CameraWindow
import numpy as np
import cv2
import yacs.config
from gaze_estimation.gaze_estimator.common import (Face, FacePartsName,
                                                   Visualizer)
from gaze_estimation.utils import load_config
from gaze_estimation import GazeEstimationMethod, GazeEstimator, GazeEstimator_ERS
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class CameraPageWindow(QtWidgets.QMainWindow,Ui_CameraWindow):
    returnSignal = pyqtSignal() #声明一个无参数的信号，可用于多个窗口之间数据传递
    def __init__(self,parent=None):
        super(CameraPageWindow, self).__init__(parent)
        self.timer_camera = QTimer() #初始化定时器
        self.timer_camera2 = QTimer() #初始化定时器2
        #self.cap = cv2.VideoCapture() #初始化摄像头
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        #self.timer_camera.start(20)
        self.gaze_flag = True
        self.CAM_NUM = 0
        self.setupUi(self)
        self.initUI()
        self.slot_init()
        self.config = load_config()  # 加载yaml配置文件
        self.demo = Demo(self.config)  # 创建demo实例
        #self.gaze()


    def initUI(self,m=True):
        #UI初始化
        if m:
            #主显示区域初始化
            main_ui = np.zeros([360,640])
            showImage = QImage(main_ui.data, main_ui.shape[1], main_ui.shape[0], QImage.Format_RGB888)
            self.cameraLabel.setPixmap(QPixmap.fromImage(showImage))

        #分区域初始化
        sub_ui = np.zeros([108, 180])
        sub_ui = QImage(sub_ui, sub_ui.shape[1], sub_ui.shape[0],
                                 QImage.Format_RGB888)
        self.right_eye_Label.setPixmap(QPixmap.fromImage(sub_ui))
        self.right_eye_Label_2.setPixmap(QPixmap.fromImage(sub_ui))
        self.right_eye_Label_3.setPixmap(QPixmap.fromImage(sub_ui))
        self.left_eye_Label.setPixmap(QPixmap.fromImage(sub_ui))
        self.left_eye_Label_2.setPixmap(QPixmap.fromImage(sub_ui))
        self.left_eye_Label_3.setPixmap(QPixmap.fromImage(sub_ui))


    def gaze_ers(self):
        #显示检测结果
        flag,show,eye_nor,pitch_list,yaw_list,maskl,maskr = self.demo.run(True)

        if flag == True:

            #脸部完整
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

            # 归一化后的图像
            eye_nor_right = eye_nor[:, 0:60][:, ::-1]  # 左右翻转右眼图像
            eye_nor_left = eye_nor[:, 60:120]

            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.cameraLabel.setPixmap(QPixmap.fromImage(showImage).scaled(640, 360))

            #####右眼#######
            #显示右眼归一化图像
            eye_nor_left = cv2.cvtColor(eye_nor_left, cv2.COLOR_GRAY2RGB)
            showImage_right = QImage(eye_nor_left,eye_nor_left.shape[1], eye_nor_left.shape[0], QImage.Format_RGB888)
            #self.left_eye_Label_2.setScaledContents(True)
            self.right_eye_Label.setPixmap(QPixmap.fromImage(showImage_right).scaled(180, 108))

            #显示右眼虹膜区域图像
            eye_right_nor_iris = (maskr[1] * 255).astype(np.uint8)
            eye_right_nor_iris = cv2.cvtColor(eye_right_nor_iris, cv2.COLOR_GRAY2RGB)
            showImage_right = QImage(eye_right_nor_iris,eye_right_nor_iris.shape[1], eye_right_nor_iris.shape[0], QImage.Format_RGB888)
            #self.left_eye_Label_2.setScaledContents(True)
            self.right_eye_Label_2.setPixmap(QPixmap.fromImage(showImage_right).scaled(180, 108))

            #显示右眼巩膜可视区域图像
            eye_right_nor_eye = (maskr[0] * 255).astype(np.uint8)
            eye_right_nor_eye = cv2.cvtColor(eye_right_nor_eye, cv2.COLOR_GRAY2RGB)
            showImage_right = QImage(eye_right_nor_eye,eye_right_nor_eye.shape[1], eye_right_nor_eye.shape[0], QImage.Format_RGB888)
            #self.left_eye_Label_2.setScaledContents(True)
            self.right_eye_Label_3.setPixmap(QPixmap.fromImage(showImage_right).scaled(180, 108))

            #####左眼#######
            # 显示左眼归一化图像
            eye_nor_right = cv2.cvtColor(eye_nor_right, cv2.COLOR_GRAY2RGB)
            showImage_left = QImage(eye_nor_right, eye_nor_right.shape[1], eye_nor_right.shape[0],
                                     QImage.Format_RGB888)
            # self.left_eye_Label_2.setScaledContents(True)
            self.left_eye_Label.setPixmap(QPixmap.fromImage(showImage_left).scaled(180, 108))

            # 显示左眼虹膜区域图像
            eye_left_nor_iris = (maskl[1] * 255).astype(np.uint8)
            eye_left_nor_iris = cv2.cvtColor(eye_left_nor_iris, cv2.COLOR_GRAY2RGB)
            showImage_left = QImage(eye_left_nor_iris, eye_left_nor_iris.shape[1], eye_left_nor_iris.shape[0],
                                     QImage.Format_RGB888)
            # self.left_eye_Label_2.setScaledContents(True)
            self.left_eye_Label_2.setPixmap(QPixmap.fromImage(showImage_left).scaled(180, 108))

            # 显示左眼巩膜可视区域图像
            eye_left_nor_eye = (maskl[0] * 255).astype(np.uint8)
            eye_left_nor_eye = cv2.cvtColor(eye_left_nor_eye, cv2.COLOR_GRAY2RGB)
            showImage_left = QImage(eye_left_nor_eye, eye_left_nor_eye.shape[1], eye_left_nor_eye.shape[0],
                                     QImage.Format_RGB888)
            # self.left_eye_Label_2.setScaledContents(True)
            self.left_eye_Label_3.setPixmap(QPixmap.fromImage(showImage_left).scaled(180, 108))

            #显示具体结果
            #self.GazetextEdit.setPlainText('pitch     yaw')

            #显示左眼视线向量
            self.GazetextEdit.setPlainText('right')
            r_eye = str(pitch_list[0])[:6]+','+str(yaw_list[0])[:6]
            self.GazetextEdit.append(r_eye)

            #显示左眼视线向量
            self.GazetextEdit.append('left')
            l_eye = str(pitch_list[1])[:6]+','+str(yaw_list[1])[:6]
            self.GazetextEdit.append(l_eye)

    def gaze_baseline(self):
        #显示检测结果
        flag,show,eye_nor,pitch_list,yaw_list,maskl,maskr = self.demo.run(True)

        if flag == True:

            #脸部完整
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

            # 归一化后的图像
            eye_nor_right = eye_nor[:, 0:60][:, ::-1]  # 左右翻转右眼图像
            eye_nor_left = eye_nor[:, 60:120]

            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.cameraLabel.setPixmap(QPixmap.fromImage(showImage).scaled(640, 360))

            #####右眼#######
            #显示右眼归一化图像
            eye_nor_left = cv2.cvtColor(eye_nor_left, cv2.COLOR_GRAY2RGB)
            showImage_right = QImage(eye_nor_left,eye_nor_left.shape[1], eye_nor_left.shape[0], QImage.Format_RGB888)
            #self.left_eye_Label_2.setScaledContents(True)
            self.right_eye_Label.setPixmap(QPixmap.fromImage(showImage_right).scaled(180, 108))

            #显示右眼虹膜区域图像
            eye_right_nor_iris = (maskr[1] * 255).astype(np.uint8)
            eye_right_nor_iris = cv2.cvtColor(eye_right_nor_iris, cv2.COLOR_GRAY2RGB)
            showImage_right = QImage(eye_right_nor_iris,eye_right_nor_iris.shape[1], eye_right_nor_iris.shape[0], QImage.Format_RGB888)
            #self.left_eye_Label_2.setScaledContents(True)
            self.right_eye_Label_2.setPixmap(QPixmap.fromImage(showImage_right).scaled(180, 108))

            #显示右眼巩膜可视区域图像
            eye_right_nor_eye = (maskr[0] * 255).astype(np.uint8)
            eye_right_nor_eye = cv2.cvtColor(eye_right_nor_eye, cv2.COLOR_GRAY2RGB)
            showImage_right = QImage(eye_right_nor_eye,eye_right_nor_eye.shape[1], eye_right_nor_eye.shape[0], QImage.Format_RGB888)
            #self.left_eye_Label_2.setScaledContents(True)
            self.right_eye_Label_3.setPixmap(QPixmap.fromImage(showImage_right).scaled(180, 108))

            #####左眼#######
            # 显示左眼归一化图像
            eye_nor_right = cv2.cvtColor(eye_nor_right, cv2.COLOR_GRAY2RGB)
            showImage_left = QImage(eye_nor_right, eye_nor_right.shape[1], eye_nor_right.shape[0],
                                     QImage.Format_RGB888)
            # self.left_eye_Label_2.setScaledContents(True)
            self.left_eye_Label.setPixmap(QPixmap.fromImage(showImage_left).scaled(180, 108))

            # 显示左眼虹膜区域图像
            eye_left_nor_iris = (maskl[1] * 255).astype(np.uint8)
            eye_left_nor_iris = cv2.cvtColor(eye_left_nor_iris, cv2.COLOR_GRAY2RGB)
            showImage_left = QImage(eye_left_nor_iris, eye_left_nor_iris.shape[1], eye_left_nor_iris.shape[0],
                                     QImage.Format_RGB888)
            # self.left_eye_Label_2.setScaledContents(True)
            self.left_eye_Label_2.setPixmap(QPixmap.fromImage(showImage_left).scaled(180, 108))

            # 显示左眼巩膜可视区域图像
            eye_left_nor_eye = (maskl[0] * 255).astype(np.uint8)
            eye_left_nor_eye = cv2.cvtColor(eye_left_nor_eye, cv2.COLOR_GRAY2RGB)
            showImage_left = QImage(eye_left_nor_eye, eye_left_nor_eye.shape[1], eye_left_nor_eye.shape[0],
                                     QImage.Format_RGB888)
            # self.left_eye_Label_2.setScaledContents(True)
            self.left_eye_Label_3.setPixmap(QPixmap.fromImage(showImage_left).scaled(180, 108))

            #显示具体结果
            #self.GazetextEdit.setPlainText('pitch     yaw')

            #显示左眼视线向量
            self.GazetextEdit.setPlainText('right')
            r_eye = str(pitch_list[0])[:6]+','+str(yaw_list[0])[:6]
            self.GazetextEdit.append(r_eye)

            #显示左眼视线向量
            self.GazetextEdit.append('left')
            l_eye = str(pitch_list[1])[:6]+','+str(yaw_list[1])[:6]
            self.GazetextEdit.append(l_eye)

    def camera(self):
        #只显示拍摄到的画面，不检测
        fram = self.demo.run(False)[:,::-1,:]
        show = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.cameraLabel.setPixmap(QPixmap.fromImage(showImage).scaled(640, 360))

    def slot_init(self):
        #self.timer_camera.timeout.connect(self.show_camera) #定时不断
            #信号和槽连接
        #self.timer_camera.timeout.connect(self.gaze)  # 定时不断
        self.pushButton_b.clicked.connect(self.control_b)
        self.pushButton_l.clicked.connect(self.control_l)
        self.pushButton_h.clicked.connect(self.control_h)
        self.cameraButton.clicked.connect(self.start_gaze)

        #self.cameraButton.clicked.connect(self.slotCameraButton) #开关摄像头
        #self.gazeButton_2.clicked.conn

    def start_gaze(self):
        #开始检测按钮
        if self.gaze_flag:
            self.timer_camera2.stop()
            self.timer_camera.start(30)
            combo_text = self.comboBox.currentText()
            if combo_text == '基于眼部区域分离':
                self.timer_camera.timeout.connect(self.gaze_ers)
            elif combo_text == 'baseline':
                self.timer_camera.timeout.connect(self.gaze_baseline)

            self.cameraButton.setText('关闭检测')
            self.gaze_flag = False
        else:
            #结束检测
            self.timer_camera.stop()
            self.initUI(m=False)
            self.timer_camera2.start(30)
            self.timer_camera2.timeout.connect(self.camera)
            self.cameraButton.setText('开启检测')
            self.gaze_flag = True

    def control_b(self):
        #脸框显示按钮
        self.demo.show_bbox = not self.demo.show_bbox
        if self.demo.show_bbox:
            self.pushButton_b.setText('关闭脸框')
        else:
            self.pushButton_b.setText('开启脸框')

    def control_l(self):
        #；脸部关键点显示按钮
        self.demo.show_landmarks = not self.demo.show_landmarks
        if self.demo.show_landmarks:
            self.pushButton_l.setText('关闭脸部关键点')
        else:
            self.pushButton_l.setText('开启脸部关键点')

    def control_h(self):
        #头部姿态显示按钮
        self.demo.show_head_pose = not self.demo.show_head_pose
        if self.demo.show_head_pose:
            self.pushButton_h.setText('关闭头部姿态')
        else:
            self.pushButton_h.setText('开启头部姿态')


    def show_camera(self):
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # self.cap.set(cv2.CAP_PROP_FPS, 30)
        #
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,960)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        # flag,self.image = self.cap.read()
        # show = cv2.resize(self.image,(480,320))
        # show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        show = self.visualizer.image
        showImage = QImage(show.data, show.shape[1],show.shape[0],QImage.Format_RGB888)
        self.cameraLabel.setPixmap(QPixmap.fromImage(showImage))

    #打开关闭摄像头控制
    def slotCameraButton(self):
        if self.timer_camera.isActive() == False:
        #打开摄像头并显示图像信息
            self.openCamera()
        else:
            #关闭摄像头并清空显示信息
            self.closeCamera()

    #打开摄像头
    def openCamera(self):
        flag = self.cap.open(self.CAM_NUM) #可用于检查初始化是否成功
        if flag == False:
            msg = QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
            buttons=QMessageBox.Ok,
            defaultButton=QMessageBox.Ok)
        else:
            self.timer_camera.start(30)
            self.cameraButton.setText('关闭摄像头')

    #关闭摄像头
    def closeCamera(self):
        self.timer_camera.stop()
        #self.cap.release()
        self.cameraLabel.clear()
        self.cameraButton.setText('打开摄像头')

class Demo:
    # ord函数返回一个字符的ASCII数值,QUIT_KEYS是一个集合，无序不重复元素序列={113, 27}
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: yacs.config.CfgNode):
        self.config = config
        #self.gaze_estimator = GazeEstimator(config) #MPIIGaze模型
        self.gaze_estimator_ERS = GazeEstimator_ERS(config)  #眼部区域分离模型
        self.visualizer = Visualizer(self.gaze_estimator_ERS.camera)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image #眼部归一化图像显示控制
        self.show_template_model = self.config.demo.show_template_model

    def run(self,flag) -> None:
        ok, frame = self.cap.read()
        if flag:
            #校正
            undistorted = cv2.undistort(
                frame, self.gaze_estimator_ERS.camera.camera_matrix,
                self.gaze_estimator_ERS.camera.dist_coefficients)

            self.visualizer.set_image(frame.copy())

            faces = self.gaze_estimator_ERS.detect_faces(undistorted) #得到脸部的各种关键信息与视线估计结果

            for face in faces:
                self.gaze_estimator_ERS.estimate_gaze(undistorted, face) #视线估计
                self._draw_face_bbox(face)
                self._draw_head_pose(face)
                self._draw_landmarks(face)
                self._draw_face_template_model(face)
                pitch_list,yaw_list = self._draw_gaze_vector(face)
                normalized = self._display_normalized_image(face).copy()
                maskl = face.leye.mask
                maskr = face.reye.mask


            if self.config.demo.use_camera:
                self.visualizer.image = self.visualizer.image[:, ::-1]
            if self.writer:
                self.writer.write(self.visualizer.image)
            if self.config.demo.display_on_screen:
                #cv2.imshow('frame', self.visualizer.image)
                show = self.visualizer.image.copy()
            if faces != []:
                return True,show,normalized,pitch_list,yaw_list,maskl,maskr
            else:
                return False,0,0,0,0,0,0
        else:
            return frame



    def _create_capture(self) -> cv2.VideoCapture:
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(2)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator_ERS.camera.width)  #设置摄像头图片大小——宽
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator_ERS.camera.height) #设置摄像头图片大小——窄
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        output_path = self.output_dir / f'{self._create_timestamp()}.{ext}'
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator_ERS.camera.width,
                                  self.gaze_estimator_ERS.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    # def _wait_key(self) -> None:
    #     key = cv2.waitKey(self.config.demo.wait_time) & 0xff
    #     if key in self.QUIT_KEYS:
    #         self.stop = True
    #     elif key == ord('b'):
    #         self.show_bbox = not self.show_bbox
    #     elif key == ord('l'):
    #         self.show_landmarks = not self.show_landmarks
    #     elif key == ord('h'):
    #         self.show_head_pose = not self.show_head_pose
    #     elif key == ord('n'):
    #         self.show_normalized_image = not self.show_normalized_image
    #     elif key == ord('t'):
    #         self.show_template_model = not self.show_template_model

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        return normalized #直接输出左右眼归一化结果
        #cv2.imshow('normalized', normalized)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        pitch_list, yaw_list = [], []
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                pitch_list.append(pitch)
                yaw_list.append(yaw)
                logger.info(
                    f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
            return pitch_list, yaw_list
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError



if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling) #自定义缩放

    app = QApplication(sys.argv)
    myWin = CameraPageWindow()
    myWin.show()
    sys.exit(app.exec_())