# 표준 라이브러리
import argparse
import asyncio
import json
import sys
import time

# 외부 라이브러리
import cv2
import qasync
import websockets
from PyQt5.QtCore import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtWidgets import QApplication
from PyQt5.uic import loadUi
import numpy as np

# 프로젝트 내의 모듈
from common.custom_class import CMainWindow
from common.function import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=7, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.set_facecolor('black')
        self.axes.tick_params(axis='both', colors='white', labelcolor='white')
        self.clear_axes()


        super(MyMplCanvas, self).__init__(fig)
    
    def clear_axes(self):
        self.axes.clear()
        self.axes.set_xlim(-35, 35)
        self.axes.set_ylim(0, 50)
        self.axes.set_facecolor('black')

        # 왼쪽과 오른쪽 spine 숨기기
        self.axes.spines['top'].set_visible(False)  # 상단 spine 숨기기
        self.axes.spines['bottom'].set_position('zero')  # 하단 spine을 0 위치로 이동
        self.axes.spines['left'].set_visible(False)  # 왼쪽 spine 숨기기
        self.axes.spines['right'].set_position('zero')  # 오른쪽 spine을 가운데로 이동
        # Y 축 눈금을 오른쪽에 표시
        self.axes.yaxis.tick_right()

        self.axes.plot([-30, 0, 30], [20, 0, 20], color='white', linestyle='--', marker='', linewidth=0.7)

class MainApp(CMainWindow):
    frame_queue = asyncio.Queue(3)
    metadata_queue = asyncio.Queue()
    task_list = []

    def __init__(self, mode):
        super().__init__()
        loadUi("./ui/main_window.ui", self)
        
        self.mpl_canvas = MyMplCanvas(self)
        self.horizontalLayout_videoLayout.addWidget(self.mpl_canvas)

        self.mode = mode
        self.pushButton_importVideo.clicked.connect(self.import_video_button_clicked)
        self.pushButton_importCamera.clicked.connect(self.import_camera_button_clicked)
        self.pushButton_startVideo.clicked.connect(self.start_video_button_clicked)
        self.pushButton_stopVideo.clicked.connect(self.stop_video_button_clicked)
        self.pushButton_restartVideo.clicked.connect(self.restart_video_button_clicked)

    @qasync.asyncSlot()
    async def import_camera_button_clicked(self):
        await self.stop_video_button_clicked()
        self.frame_queue = asyncio.Queue(30)
        self.metadata_queue = asyncio.Queue()
        self.cap = cv2.VideoCapture(0)
        self.file_name = 'Camera'
        if self.cap:
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.video_height_ratio = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 640

            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / self.video_height_ratio)
            self.video_height = 640
            video_info_text = f"\nvideo size : {self.video_width} x {self.video_height}"
            self.plainTextEdit_videoInfo.setPlainText(video_info_text)

            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.video_width, self.video_height))
                self.label_videoWindow.setPixmap(frame_to_pixmap(frame))            
        else:
            show_message(self, "카메라가 존재하지 않습니다.")
            return
        
    @qasync.asyncSlot()
    async def import_video_button_clicked(self):
        # file_name, _ = QFileDialog.getOpenFileName(self, "영상 불러오기", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        await self.stop_video_button_clicked()
        self.frame_queue = asyncio.Queue(30)
        self.metadata_queue = asyncio.Queue()
        file_name, _ = QFileDialog.getOpenFileName(self, "영상 불러오기", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.file_name = file_name

            self.cap = cv2.VideoCapture(self.file_name)
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            # self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height_ratio = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 640

            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / self.video_height_ratio)
            self.video_height = 640
            video_info_text = f"\n파일 이름 : \n{self.file_name} \n\nfps : {self.fps} \n\nvideo size : {self.video_width} x {self.video_height}"
            self.plainTextEdit_videoInfo.setPlainText(video_info_text)

            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.video_width, self.video_height))
                self.label_videoWindow.setPixmap(frame_to_pixmap(frame))            
        else:
            show_message(self, "파일이 존재하지 않습니다.")
            return
        
    @qasync.asyncSlot()
    async def restart_video_button_clicked(self):
        await self.stop_video_button_clicked()
        self.frame_queue = asyncio.Queue(30)
        self.metadata_queue = asyncio.Queue()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        await self.start_video_button_clicked()
    
    @qasync.asyncSlot()
    async def stop_video_button_clicked(self):
        for task in self.task_list:
            task.cancel()

    @qasync.asyncSlot()
    async def start_video_button_clicked(self):
        file_name = getattr(self, 'file_name', None)
        if not file_name:
            show_message(self, "파일을 먼저 선택해주세요!")
            return

        server_uri1 = "ws://10.28.224.52:30300/ws"
        server_uri2 = "ws://10.28.224.52:30301/ws"

        server_uri1 = "ws://10.28.224.34:30348/ws"
        server_uri2 = "ws://10.28.224.34:30349/ws"

        async with websockets.connect(server_uri1) as websocket1, websockets.connect(server_uri2) as websocket2:
            self.send_task = asyncio.create_task(self.send_frame(websocket1, websocket2, self.cap))
            self.receive_task1 = asyncio.create_task(self.receive_frame(websocket1))
            self.receive_task2 = asyncio.create_task(self.receive_frame(websocket2))
            self.render_task = asyncio.create_task(self.render_frame((self.video_height, self.video_width), self.fps))

            self.task_list.append(self.send_task)
            self.task_list.append(self.receive_task1)
            self.task_list.append(self.receive_task2)
            self.task_list.append(self.render_task)
            await asyncio.gather(
                self.send_task, 
                self.receive_task1, 
                self.receive_task2, 
                self.render_task,
                return_exceptions=True  
            )


    def update_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label_videoWindow.setPixmap(QPixmap.fromImage(qt_image))
    
    def draw_bboxes(self, frame, bboxes, frame_shape):
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            class_id = bbox[4]
            median_point = bbox[5]
            max_point = bbox[6]                       
            mean_point = bbox[7] 
            middle_point = bbox[8]
            # 원래 프레임의 크기에 맞게 bbox 좌표 조정
            x1 = int(x1 * frame_shape[1] / 640)
            y1 = int(y1 * frame_shape[0] / 640)
            x2 = int(x2 * frame_shape[1] / 640)
            y2 = int(y2 * frame_shape[0] / 640)
            #detections = sv.Detections(np.array([x1,y1,x2,y2]))
            if class_id == 9:
                mosaic_area = frame[y1:y2, x1:x2]
                X, Y = x1//30, y1//30
                if X <= 0:
                    X = 1
                if Y <= 0:
                    Y = 1
                mosaic_area = cv2.resize(mosaic_area, (X,Y))
                mosaic_area = cv2.resize(mosaic_area, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                frame[y1:y2, x1:x2] = mosaic_area
            # if max_point > 10:
                '''corner_annotator = sv.BoxCornerAnnotator()
                frame = corner_annotator.annotate(
                scene=frame.copy(),
                detections=detections
                )'''
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"Class ID: {class_id}, Max Point: {int(max_point)}, Middle Point: {int(middle_point)}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    async def send_frame(self, websocket1, websocket2, cap):
        frame_count = 0
        start_time = time.time()
        while cap.isOpened():
            current_time = time.time() - start_time
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            #print(f'current_time: {current_time}')
            #print(f'video_time: {video_time}')
            ret, frame = cap.read()
            if (current_time - video_time) > 1:
                continue
            if ret == None:
                break
            try:
                send_frame = cv2.resize(frame, (640, 640))
                frame = cv2.resize(frame, (self.video_width, self.video_height))
            except:
                raise Exception("영상이 끝났습니다!")
            
            await self.frame_queue.put(frame)
            _, img_encoded = cv2.imencode('.jpg', send_frame)
            if frame_count % 2 == 0:
                await websocket1.send(img_encoded.tobytes())
            else:
                await websocket2.send(img_encoded.tobytes())
            frame_count += 1

    async def receive_frame(self, websocket):
        while True:
            result = await websocket.recv()
            result_dict = json.loads(result)
            bboxes = result_dict.get("bboxes", [])
            await self.metadata_queue.put(bboxes)

    async def render_frame(self, frame_shape, fps):
        FPS = fps
        time_per_frame = 1 / fps
        prev_time = time.time() - 1
        fps_time = time.time()
        frame_count = 0
        while True:
            metadata = await self.metadata_queue.get()
            frame = await self.frame_queue.get()
            self.draw_bboxes(frame, metadata, frame_shape)

            curr_time = time.time()
            elapsed = curr_time - prev_time

            if 1 < (time.time() - fps_time):
                FPS = frame_count
                frame_count = 0
                fps_time = time.time()

            cv2.putText(frame, f"FPS: {FPS}", (frame.shape[1] - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            if elapsed < time_per_frame:
                await asyncio.sleep(time_per_frame - elapsed)

            # cv2.imshow("Object Detection", frame)
            self.update_frame(frame)
            self.update_figure(metadata)
            prev_time = time.time()

            frame_count += 1

    def update_figure(self, metadata, figure_size=[70, 70]):
        self.mpl_canvas.clear_axes()
        for bbox in metadata:
            x1, y1, x2, y2 = bbox[:4]
            class_id = bbox[4]
            median_point = bbox[5]
            max_point = bbox[6]                     
            mean_point = bbox[7] 
            middle_point = bbox[8]
            if class_id != 9:
                x = int(((x1 + x2)/2) * figure_size[1] / 640) - 35
                # max point에 따라 log로 0~70 out
                y = 70 - np.log(max_point + 1) * (70 / np.log(256))
                # max point에 따라 선형으로 0~70 out
                # y = 70 * (1 - max_point / 255)
                y += int(abs(x) * (2/3))
                self.mpl_canvas.axes.plot(x, y, linestyle='none', marker='*', color="red", alpha=1)
            
        self.mpl_canvas.draw()
def main():
    parser = argparse.ArgumentParser(description='Program Mode')
    parser.add_argument('--mode', type=str, help='DEV or PROD', default="DEV")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    main_window = MainApp(args.mode)
    main_window.show()

    with loop:
        loop.run_forever()

if __name__ == '__main__':
    main()