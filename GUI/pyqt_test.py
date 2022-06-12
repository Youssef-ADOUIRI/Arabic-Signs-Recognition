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

#Tensorflow utils
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join( ROOT_DIR , 'saved_model/ARS_REC_model_gray_v3.h5') 
model = models.load_model(path)

IMG_SIZE = 64
CATEGORIES = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 
             'gaaf', 'ghain', 'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 
             'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 'ta', 
             'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay']

fps = FPS().start()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray,int)

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
                
                cv2.rectangle(cv_img , (300,300) , (100,100), (0,255,0) , 0)
        
                image_to_process = cv_img[100:300, 100:300]
                fps.update()
                g_img = cv2.cvtColor(image_to_process,cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(g_img, (IMG_SIZE , IMG_SIZE))
                l_img = [resized]

                input = np.array(l_img , dtype=np.float32).reshape(-1 , IMG_SIZE, IMG_SIZE , 1 )
                #converting value from [0,255] to [0,1], then predict
                input /= 255.0
                prediction = model.predict(input)
                ind = np.argmax(prediction)

                self.change_pixmap_signal.emit(cv_img ,ind)
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
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)
        self.image_label.setPixmap(self.grey)
        self.Vid_thread = VideoThread()
        self.Vid_thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.Vid_thread.start()
        
    @pyqtSlot(np.ndarray ,int)
    def update_image(self, cv_img , ind):
        qt_img = self.convert_cv_qt(cv_img)
        print('prediction : ' , CATEGORIES[ind])
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def predict_img(self,image):
        g_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized = imutils.resize(g_img, width=IMG_SIZE , height=IMG_SIZE)
        l_img = [resized]

        input = np.array(l_img , dtype=np.float32).reshape(-1 , IMG_SIZE, IMG_SIZE , 1 )
        #converting value from [0,255] to [0,1], then predict
        input /= 255.0
        prediction = model.predict(input)
        ind = np.argmax(prediction)
        #print(ind[0][0])
        return ind
    
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
    # create the instance of our Window
    window = Window()
    window.show()

    sys.exit(App.exec())