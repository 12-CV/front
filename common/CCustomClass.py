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
        
        self.pushButton_applyConfig: QPushButton
        self.pushButton_importVideo: QPushButton