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
        self.view = graphicsView_mainViewer(self)
        self.videoLayout.addWidget(self.view)
        
        self.pushButton_importVideo.clicked.connect(self.import_video_button_clicked)
        self.plainTextEdit_mainLogger.setPlainText("[System] 영상을 입력하세요. ")

    def import_video_button_clicked(self):
        self.file_name, _ = QFileDialog.getOpenFileName(self, "영상 불러오기", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if self.file_name:
            cap = cv2.VideoCapture(self.file_name)
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.view.viewer_width = self.video_width
            self.view.viewer_height = self.video_height

            video_info_text = f"\n파일 이름 : \n{self.file_name} \n\nfps : {self.fps} \n\nvideo size : {self.video_width} x {self.video_height}"
            self.plainTextEdit_videoInfo.setPlainText(video_info_text)
            
            self.view.video_play(self.file_name)
            
        else:
            show_message(self, "파일 불러오기에 실패했습니다.")

class graphicsView_mainViewer(QGraphicsView):
    def __init__(self, parent):
        super(graphicsView_mainViewer, self).__init__(parent)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.file_name = ""
        self.gScene = QGraphicsScene(self)
        self.mediaPlayer = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self.videoItem = QGraphicsVideoItem()

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.viewer_width = 1600
        self.viewer_height = 900
        self.videoItem.setPos(0, 0)
        self.videoItem.setSize(QSizeF(self.viewer_width, self.viewer_height))
        self.gScene.addItem(self.videoItem)


        # self.videoItem.setPos(0,0)

        # self.setScene(self.gScene)
        
    def video_play(self, file):
        self.file_name = file
        video = QMediaContent((QUrl.fromLocalFile(file)))
        self.mediaPlayer.setVideoOutput(self.videoItem)
        self.setScene(self.gScene)
        self.mediaPlayer.setMedia(video)
        self.mediaPlayer.play()
        # self.fitInView(self.gScene.sceneRect(), Qt.KeepAspectRatio)


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