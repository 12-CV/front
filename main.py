import argparse
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from common.custom_class import CMainWindow
from common.function import *
from controllers.main_window.logic import *

class MainApp(CMainWindow):
    def __init__(self, mode):
        super().__init__()
        loadUi("./ui/main_window.ui", self)
        self.pushButton_importVideo.clicked.connect(self.import_video_button_clicked)

    def import_video_button_clicked(self):
        self.file_name, _ = QFileDialog.getOpenFileName(self, "영상 불러오기", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if self.file_name:
            self.plainTextEdit_videoInfo.setPlainText(f"파일 이름 : {self.file_name}")
        else:
            show_message(self, "파일 불러오기에 실패했습니다.")
 
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