from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QColor, QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
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


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
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
  
    # create the instance of our Window
    window = Window()
    window.show()

    # start the app
    sys.exit(App.exec())