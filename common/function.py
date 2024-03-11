from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
import cv2

def show_message(program, message):
    QMessageBox.information(program, "Alert", message)

def frame_to_pixmap(frame):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_image)