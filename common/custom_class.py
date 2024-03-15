from PyQt5.QtWidgets import *

class CMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.checkBox_availableRouteDetect: QCheckBox
        self.checkBox_collisionDetect: QCheckBox
        self.checkBox_mosaicFace: QCheckBox
        self.checkBox_objectDetect: QCheckBox

        self.plainTextEdit_mainLogger: QPlainTextEdit
        self.plainTextEdit_videoInfo: QPlainTextEdit
        
        self.pushButton_startVideo: QPushButton
        self.pushButton_stopVideo: QPushButton
        self.pushButton_restartVideo: QPushButton
        self.pushButton_importVideo: QPushButton
        self.pushButton_importCamera: QPushButton
        
        self.graphicsView_mainViewer: QGraphicsView
        self.label_videoWindow: QLabel