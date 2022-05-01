from cProfile import label
from tkinter import font
from PyQt5.QtWidgets import *
<<<<<<< HEAD
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5 import QtGui , QtCore
=======
from PyQt5.QtGui import QPixmap, QColor, QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
>>>>>>> gui
import sys
import cv2
import numpy as np


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


styleSheet = """
    QWidget{
        background: #313131;
    }
    QLabel{
        padding-top: 5px;
        text-align: center;
        text-transform: uppercase;
        font-weight: 200;
        font-size: 2em;
        color: rgb(238, 238, 238)
    }
"""

LxSize = 200
LySize = 60

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
<<<<<<< HEAD
        #layout of the window
        self.layout = QVBoxLayout(self)
        # set the title
        self.setWindowTitle("ASR")
        self.setWindowIcon(QtGui.QIcon('logo192.png'))
    
        # set the geometry
        self.setGeometry(10, 60, 1280+10, 720+60)

        # create label widget to display content on screen 
        self.label = QLabel("Arabic Sign Language", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        #self.label.setGeometry((1280 - LxSize)//2 , (100 - LySize)//2 ,LxSize ,LySize)
        self.label.setGeometry(0 , 0 ,1280 ,100)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setFont( QtGui.QFont('Fira Sans' , 30) )
        self.layout.addWidget(self.label)
        
        self.available_cameras = QCameraInfo.availableCameras()
        # if no camera found
        if not self.available_cameras:
            # exit the code
            print('No available camera !!')
            sys.exit()
        
        self.viewfinder  = QCameraViewfinder()
        
        self.viewfinder.show()
        self.select_camera(0)
        self.layout.addWidget(self.viewfinder)

        # show all the widgets
        self.show()
    
    def select_camera(self, i):
  
        # getting the selected camera
        self.camera = QCamera(self.available_cameras[i])
  
        # setting view finder to the camera
        self.camera.setViewfinder(self.viewfinder)
  
        # setting capture mode to the camera
        self.camera.setCaptureMode(QCamera.CaptureStillImage)
  
        # if any error occur show the alert
        self.camera.error.connect(lambda: self.alert(self.camera.errorString()))
  
        # start the camera
        self.camera.start()
  
        # creating a QCameraImageCapture object
        self.capture = QCameraImageCapture(self.camera)
  
        # showing alert if error occur
        self.capture.error.connect(lambda error_msg, error,
                                   msg: self.alert(msg))
  
        # when image captured showing message
        self.capture.imageCaptured.connect(lambda d,
                                           i: self.status.showMessage("Image captured : " 
                                                                      + str(self.save_seq)))
  
        # getting current camera name
        self.current_camera_name = self.available_cameras[i].description()
  
        # initial save sequence
        self.save_seq = 0


# create pyqt5 app
App = QApplication(sys.argv)
App.setStyleSheet(styleSheet)
window = Window()
=======
        self.setWindowTitle("Qt static label demo")
        self.display_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Demo')

        # create a vertical box layout and add the two labels
        wid = QWidget(self)
        self.setCentralWidget(wid)
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        wid.setLayout(vbox)
        
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    # create pyqt5 app
    App = QApplication(sys.argv)
>>>>>>> gui
  
    # create the instance of our Window
    window = Window()
    window.show()

    # start the app
    sys.exit(App.exec())