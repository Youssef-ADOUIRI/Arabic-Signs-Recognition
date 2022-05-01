from cProfile import label
from tkinter import font
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5 import QtGui , QtCore
import sys

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
  
# start the app
sys.exit(App.exec())