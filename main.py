import argparse
import cv2
import sys

from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtCore import *

from controllers.main_window.logic import *
from common.function import *
from common.custom_class import CMainWindow

class MainApp(CMainWindow):
    def __init__(self, mode):
        super().__init__()

        loadUi("./ui/main_window.ui", self)
        
        self.pushButton_importVideo.clicked.connect(self.import_video_button_clicked)
        self.plainTextEdit_mainLogger.setPlainText("[System] 영상을 입력하세요. ")

    def import_video_button_clicked(self):
        # file_name, _ = QFileDialog.getOpenFileName(self, "영상 불러오기", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        file_name, _ = QFileDialog.getOpenFileName(self, "영상 불러오기", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.file_name = file_name

            cap = cv2.VideoCapture(self.file_name)
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_info_text = f"\n파일 이름 : \n{self.file_name} \n\nfps : {self.fps} \n\nvideo size : {self.video_width} x {self.video_height}"
            self.plainTextEdit_videoInfo.setPlainText(video_info_text)
        else:
            show_message(self, "파일이 존재하지 않습니다.")
            return

def main():
    parser = argparse.ArgumentParser(description='Program Mode')
    parser.add_argument('--mode', type=str, help='DEV or PROD', default="DEV")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    main_window = MainApp(args.mode)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()