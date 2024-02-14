import argparse
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

class MainApp(QMainWindow):
    def __init__(self, mode):
        super().__init__()
        loadUi("ui/untitled.ui", self)


def main():
    parser = argparse.ArgumentParser(description='test parser')
    parser.add_argument('--example', type=str, help='example', default="nothing")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    main_window = MainApp(args)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()