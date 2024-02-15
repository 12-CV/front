import argparse
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from common.CCustomClass import CMainWindow

class MainApp(CMainWindow):
    def __init__(self, mode):
        super().__init__()
        loadUi("ui/main_window.ui", self)


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