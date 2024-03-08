import argparse
import cv2
import sys
import time
from threading import Lock
import os

from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QImage, QPixmap

from controllers.main_window.logic import *
from common.function import *
from common.custom_class import CMainWindow
from controllers.video.controller import *


class MainApp(CMainWindow):
    def __init__(self, mode):
        super().__init__()
        
        loadUi("./ui/main_window.ui", self)

        self.pushButton_importVideo.clicked.connect(self.import_video_button_clicked)
        self.pushButton_startVideo.clicked.connect(self.start_video_button_clicked)

    def import_video_button_clicked(self):
        # file_name, _ = QFileDialog.getOpenFileName(self, "영상 불러오기", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        file_name, _ = QFileDialog.getOpenFileName(self, "영상 불러오기", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.file_name = file_name

            self.cap = cv2.VideoCapture(self.file_name)
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_info_text = f"\n파일 이름 : \n{self.file_name} \n\nfps : {self.fps} \n\nvideo size : {self.video_width} x {self.video_height}"
            self.plainTextEdit_videoInfo.setPlainText(video_info_text)

            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640,640))
                self.label_videoWindow.setPixmap(frame_to_pixmap(frame))            
        else:
            show_message(self, "파일이 존재하지 않습니다.")
            return

    def start_video_button_clicked(self):
        file_name = getattr(self, 'file_name', None)
        if not file_name:
            show_message(self, "파일을 먼저 선택해주세요!")
            return
        
        # 비디오를 처음으로 돌림
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_dict = {}
        frame_dict_lock = Lock()
        frame_metadata_queue = queue.Queue()
        finish_event = threading.Event()

        video_send_thread = VideoSendThread(self.cap, frame_dict, frame_dict_lock, finish_event)
        video_send_thread.start()

        video_recieve_thread = VideoRecieveThread(frame_metadata_queue, finish_event)
        video_recieve_thread.start()

        video_render_thread = VideoRenderThread(frame_dict, frame_dict_lock, frame_metadata_queue, finish_event)
        video_render_thread.video_signal.connect(self.update_frame)
        video_render_thread.start()
        time.sleep(0.01)

    def update_frame(self, image:QImage):
        self.label_videoWindow.setPixmap(QPixmap.fromImage(image))

        
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