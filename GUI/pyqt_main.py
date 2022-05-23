from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QColor, QImage , QIcon
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from qtwidgets import Toggle, AnimatedToggle
import sys
import cv2
import numpy as np
import os
from tensorflow.keras import models
import imutils
from imutils.video import FPS
from threading import Thread
import arabic_reshaper
from bidi.algorithm import get_display
import time

'''
#Python 3
if sys.version_info >= (3, 0):
	from queue import Queue
#Python 2.7
else:
	from Queue import Queue
'''


#Tensorflow utils
ROOT_DIR =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join( ROOT_DIR , 'saved_model/ARS_REC_model_gray_v3.h5') 
model = models.load_model(path)
IMG_SIZE = 64
CATEGORIES = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 
             'gaaf', 'ghain', 'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 
             'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 'ta', 
             'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay']
#Unicode arabic notations
buckwalterMod = {
        'ء': 'c', 'ا': 'A', 'إ': 'A',
        'أ': 'aleff', 'آ': 'A', 'ب': 'bb',
        'ة': 'toot', 'ت': 'taa', 'ث': 'thaa',
        'ج': 'jeem', 'ح': 'haa', 'خ': 'khaa',
        'د': 'dal', 'ذ': 'thal', 'ر': 'ra',
        'ز': 'zay', 'س': 'seen', 'ش': 'sheen',
        'ص': 'saad', 'ض': 'dhad', 'ط': 'ta',
        'ظ': 'dha', 'ع': 'ain', 'غ': 'ghain',
        'ف': 'fa', 'ق': 'gaaf', 'ك': 'kaaf',
        'ل': 'laam', 'م': 'meem', 'ن': 'nun',
        'ه': 'ha', 'ؤ': 'c', 'و': 'waw',
        'ى': 'yaa', 'ئ': 'c', 'ي': 'ya',
        }

reversedBucket = {y: x for x, y in buckwalterMod.items()} 

fps = FPS().start()
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        #self.Q = Queue(maxsize=128)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            #if not self.Q.full():
            ret, cv_img = cap.read()
            if ret:
                #self.Q.put(cv_img)
                self.change_pixmap_signal.emit(cv_img)
            else:
                self.stop()
                return
        # shut down capture system
        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class Window(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASLR GUI")
        self.setWindowIcon(QIcon('logo192.png'))
        self.resize(662,720)
        self.display_width = 640
        self.display_height = 480
        self.grey = QPixmap(self.display_width, self.display_height)
        self.grey.fill(QColor('darkGray'))
        #title
        self.title = QLabel('Arabic Signs Language')
        self.title.setObjectName('title1')
        self.title.setAlignment(Qt.AlignCenter)
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.setObjectName('vid')
        self.image_label.resize(self.display_width, self.display_height)
        self.image_label.setPixmap(self.grey)
        # create a text label
        predi = 'none'
        self.textLabel = QLabel(predi , self)
        self.textLabel.setObjectName('predi')
        self.textLabel.setAlignment(Qt.AlignCenter)
        arabChar = '\u0626'
        self.arabicNotation = QLabel(arabChar , self)
        self.textLabel.setObjectName('arabNot')
        self.arabicNotation.setAlignment(Qt.AlignCenter)

        #self.phrase = QLabel(phrase_txt , self)
        self.btn_openCam = QPushButton('Open camera', self)
        self.btn_openCam.clicked.connect(self.openCamera_click)
        self.btn_openCam.setCheckable(True)

        self.btn_predction = QPushButton('Give prediction', self)
        self.btn_predction.clicked.connect(self.takePrediction)

        self.toggle_1 = Toggle()
        # create a vertical box layout and add the labels
        wid = QWidget(self)
        self.setCentralWidget(wid)
        vbox = QGridLayout()
        vbox.addWidget(self.title , 0,1)
        vbox.addWidget(self.image_label,1,1 )
        vbox.addWidget(self.textLabel,2,1)
        #vbox.addWidget(self.toggle_1,2,2)
        vbox.addWidget(self.btn_openCam, 3 , 1 )
        vbox.addWidget(self.btn_predction , 4, 1 )
        vbox.addWidget(self.arabicNotation , 5 , 1)
        #vbox.addWidget(self.phrase)
        
        # set the vbox layout as the widgets layout
        wid.setLayout(vbox)
        self.Vid_thread = None

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv2.rectangle(cv_img , (300,300) , (100,100), (0,255,0) , 0))
        image_to_process = cv_img[100:300, 100:300]
        fps.update()
        index = self.predict_img(image_to_process)
        prediction = CATEGORIES[index]
        #currentVal = reversedBucket[prediction]
        self.textLabel.setText(prediction)
        #arabNot = ord(reversedBucket[prediction]).encode('ascii', 'backslashreplace').decode("utf-8")
        self.arabicNotation.setText(reversedBucket[prediction])
        self.image_label.setPixmap(qt_img)
        if not self.btn_openCam.isChecked() or self.Vid_thread is None:
            self.image_label.setPixmap(self.grey)
        

    def predict_img(self,image):
        g_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized = imutils.resize(g_img, width=IMG_SIZE , height=IMG_SIZE)
        l_img = [resized]

        input = np.array(l_img , dtype=np.float32)
        input = input.reshape(-1 , IMG_SIZE, IMG_SIZE , 1 )
        #converting value from [0,255] to [0,1], then predict
        prediction = model.predict(input/255.0)
        ind = np.argmax(prediction)
        #print(ind[0][0])
        return ind

    
    @pyqtSlot()
    def openCamera_click(self):
        if self.btn_openCam.isChecked():
            self.btn_openCam.setText("Close camera")
            # create the video capture thread
            self.Vid_thread = VideoThread()
            # connect its signal to the update_image slot
            self.Vid_thread.change_pixmap_signal.connect(self.update_image)
            # start the thread
            self.Vid_thread.start()

        elif self.Vid_thread is not None:
            self.btn_openCam.setText("Open camera")
            self.Vid_thread.stop()
            # set the image image to the grey pixmap
            self.image_label.setPixmap(self.grey)
    @pyqtSlot()
    def takePrediction(self):
        print("Prediction is : ",self.textLabel.text() )

        
    
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def closeEvent(self, event):
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        if self.Vid_thread:
            self.Vid_thread.stop()
        event.accept()
    





if __name__ == "__main__":
    # create pyqt5 app
    # start the app
    App = QApplication(sys.argv)
    
    qss = """
        QWidget{
            margin: 6px;
            padding: 0;
        }
        *{
            background-color: #202020 ;
        }
        QLabel{
            color: #FFFFFF;
        }
        QPushButton{
            color: #FFDF6C;
            background-color:#707070;
            padding: 5px
        }
    """
    #494D5F
    # create the instance of our Window
    window = Window()
    window.setStyleSheet(qss)
    window.arabicNotation.setStyleSheet("text-align: center;font-size: 30px;")
    window.textLabel.setStyleSheet("text-align: center;font-size: 30px;")
    window.title.setStyleSheet("text-align: center;font-size: 40px; color:#FFDF6C")
    window.image_label.setStyleSheet("background:darkGray;border-top-left-radius: 30px;border-top-right-radius: 30px;")
    window.show()
    

    sys.exit(App.exec())