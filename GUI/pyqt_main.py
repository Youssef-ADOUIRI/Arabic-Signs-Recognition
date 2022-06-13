from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QColor, QImage , QIcon
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from qtwidgets import Toggle, AnimatedToggle
import sys
import cv2
import numpy as np
import os
from tensorflow.keras import models
from imutils.video import FPS


MODEL_NAME = 'ARS_REC_model_gray_v3.h5'
INPUT_CHANNELS = 1

#Tensorflow utils
ROOT_DIR =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join( ROOT_DIR , 'saved_model/' + MODEL_NAME) 
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
reversedBucket['la'] = 'ل'

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
                if (INPUT_CHANNELS != 3):
                    image_to_process = cv2.cvtColor(image_to_process,cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(image_to_process, (IMG_SIZE , IMG_SIZE))
                l_img = [resized]
                input = np.array(l_img , dtype=np.float32).reshape(-1 , IMG_SIZE, IMG_SIZE , INPUT_CHANNELS )
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
        self.title.setAlignment(Qt.AlignCenter)
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)
        self.image_label.setFixedWidth(self.display_width)
        self.image_label.setPixmap(self.grey)
        # create a text label
        predi = 'none'
        self.textLabel = QLabel(predi , self)
        self.textLabel.setAlignment(Qt.AlignCenter)
        

        #self.phrase = QLabel(phrase_txt , self)
        self.btn_openCam = QPushButton('Open camera', self)
        self.btn_openCam.clicked.connect(self.openCamera_click)
        self.btn_openCam.setCheckable(True)


        arabChar = 'لا شيئ'
        self.arabicNotation = QLabel(arabChar , self)
        self.arabicNotation.setAlignment(Qt.AlignCenter)
        #phrase to build
        arabic_phrase = 'اهلا بك الى البرنامج'
        self.phrase_label = QLabel(arabic_phrase , self)
        self.phrase_label.setAlignment(Qt.AlignCenter)

        #add the current prediction to phrase
        self.btn_predction = QPushButton('Add', self)
        self.btn_predction.clicked.connect(self.add_to_phrase)

        #Clear
        self.btn_constract = QPushButton('Clear' , self)
        self.btn_constract.clicked.connect(self.clear_phrase)

        # create a vertical box layout and add the labels
        wid = QWidget(self)
        self.setCentralWidget(wid)
        vbox = QGridLayout()
        vbox.addWidget(self.title , 0,1)
        vbox.addWidget(self.image_label,1,1 )

        predictionLabelHlayout = QHBoxLayout()
        predictionLabelHlayout.addWidget(self.textLabel)
        predictionLabelHlayout.addWidget(self.arabicNotation)
        vbox.addLayout(predictionLabelHlayout , 2 , 1)

        vbox.addWidget(self.btn_openCam, 3 , 1 )

        phraseBtnsHLayout = QHBoxLayout()
        phraseBtnsHLayout.addWidget(self.btn_predction)
        phraseBtnsHLayout.addWidget(self.btn_constract)

        vbox.addLayout(phraseBtnsHLayout , 4, 1 )
        
        vbox.addWidget(self.phrase_label , 5 , 1)
        
        # set the vbox layout as the widgets layout
        wid.setLayout(vbox)
        self.Vid_thread = None

    @pyqtSlot(np.ndarray , int)
    def update_image(self, cv_img , index):
        cv2.rectangle(cv_img , (300,300) , (100,100), (0,255,0) , 0)
        qt_img = self.convert_cv_qt(cv_img)    
        prediction = CATEGORIES[index]
        self.textLabel.setText(prediction)
        self.arabicNotation.setText(reversedBucket[prediction])
        self.image_label.setPixmap(qt_img)
        if not self.btn_openCam.isChecked() or self.Vid_thread is None:
            self.image_label.setPixmap(self.grey)
            self.arabicNotation.setText('لا شيئ')
            self.textLabel.setText('none')
        


    
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
    def clear_phrase(self):
        self.phrase_label.setText('') #clear the text

    @pyqtSlot()
    def add_to_phrase(self):
        if 'لا شيئ' != self.arabicNotation.text() :
            self.phrase_label.setText(self.phrase_label.text() + self.arabicNotation.text())

        
    
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
            padding: 6px
        }
    """
    #494D5F
    # create the instance of our Window
    window = Window()
    window.setStyleSheet(qss)
    window.arabicNotation.setStyleSheet("text-align: center;font-size: 30px;")
    window.textLabel.setStyleSheet("text-align: center;font-size: 30px;")
    window.title.setStyleSheet("text-align: center;font-size: 40px; color:#FFDF6C")
    window.image_label.setStyleSheet("background:darkGray;border-radius: 30px;")
    window.show()
    

    sys.exit(App.exec())