import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random

class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=7, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.set_facecolor('black')
        self.axes.tick_params(axis='both', colors='white', labelcolor='white')
        self.clear_axes()

        super(MyMplCanvas, self).__init__(fig)
    
    def clear_axes(self):
        self.axes.clear()
        self.axes.set_xlim(-35, 35)
        self.axes.set_ylim(0, 70)
        self.axes.set_facecolor('black')

        # 왼쪽과 오른쪽 spine 숨기기
        self.axes.spines['top'].set_visible(False)  # 상단 spine 숨기기
        self.axes.spines['bottom'].set_position('zero')  # 하단 spine을 0 위치로 이동
        self.axes.spines['left'].set_visible(False)  # 왼쪽 spine 숨기기
        self.axes.spines['right'].set_position('zero')  # 오른쪽 spine을 가운데로 이동
        # Y 축 눈금을 오른쪽에 표시
        self.axes.yaxis.tick_right()

        self.axes.plot([-30, 0, 30], [30, 0, 30], color='white', linestyle='--', marker='', linewidth=0.7)

class TrackingObject():
    current_location:tuple[int, int]
    prev_locations:list[tuple[int, int]]
    marker:str
    class_name: str

class MyApp(QMainWindow):
    x = [1]
    y = [10]
    def __init__(self):
        super().__init__()

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        self.mpl_canvas = MyMplCanvas(self.main_widget, width=7, height=7, dpi=100)
        layout.addWidget(self.mpl_canvas)

        self.timer = QTimer()
        self.timer.setInterval(100)# ms
        self.timer.timeout.connect(self.update_figure)
        self.timer.start()

    def update_figure(self):
        print("업데이트합니다.")
        self.mpl_canvas.clear_axes()

        self.mpl_canvas.axes.plot(self.x, self.y, linestyle='none', marker='*', alpha=0.5)
        self.y[0] += 1
        self.mpl_canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MyApp()
    main_app.show()
    sys.exit(app.exec_())