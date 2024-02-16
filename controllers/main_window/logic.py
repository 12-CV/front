from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMainWindow
from common.CCustomClass import CMainWindow


# class MainWindow(QMainWindow):

#     def __init__(self, main_window):
#         super().__init__()
#         self.main_window = main_window
#         self.file_name = ""

def import_video():
    print("import")
    file_name, _ = QFileDialog.getOpenFileName("영상 불러오기", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
    # if file_name:
    #     self.file_name = file_name
    
    return file_name


def print_filename():
    pass
