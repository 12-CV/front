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
import math

# 프로젝트 내의 모듈
from common.custom_class import CMainWindow
from common.function import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Circle


def center_square(image):
    # 이미지 크기 확인
    height, width, _ = image.shape

    # 가로와 세로 중 짧은 길이 결정
    min_dim = min(height, width)

    # 정사각형으로 이미지 자르기
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2
    end_x = start_x + min_dim
    end_y = start_y + min_dim
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image

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
        self.axes.set_xlim(-10, 10)
        self.axes.set_ylim(0, 20)
        self.axes.set_facecolor('black')

        # 왼쪽과 오른쪽 spine 숨기기
        self.axes.spines['top'].set_visible(False)  # 상단 spine 숨기기
        self.axes.spines['bottom'].set_position('zero')  # 하단 spine을 0 위치로 이동
        self.axes.spines['left'].set_visible(False)  # 왼쪽 spine 숨기기
        self.axes.spines['right'].set_position('zero')  # 오른쪽 spine을 가운데로 이동
        # Y 축 눈금을 오른쪽에 표시
        self.axes.yaxis.tick_right()

        # 중점으로부터 5, 10 거리의 위험 반경 표시
        theta = np.linspace(0, 2*np.pi, 100)
        x = 5 * np.cos(theta)
        y = 5 * np.sin(theta)
        self.axes.plot(x, y, color='white', linestyle='--', marker='', linewidth=0.7)

        x = 10 * np.cos(theta)
        y = 10 * np.sin(theta)
        self.axes.plot(x, y, color='white', linestyle='--', marker='', linewidth=0.7)
        # self.axes.plot([-30, 0, 30], [20, 0, 20], color='white', linestyle='--', marker='', linewidth=0.7)

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
            # self.video_height_ratio = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 640

            # 비디오 정사각 고정
            self.video_width = 640
            self.video_height = 640
            video_info_text = f"\nvideo size : {self.video_width} x {self.video_height}"
            self.plainTextEdit_videoInfo.setPlainText(video_info_text)

            ret, frame = self.cap.read()
            frame = center_square(frame)
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
            # self.video_height_ratio = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 640

            # 비디오 정사각 고정
            self.video_width = 640
            self.video_height = 640
            video_info_text = f"\n파일 이름 : \n{self.file_name} \n\nfps : {self.fps} \n\nvideo size : {self.video_width} x {self.video_height}"
            self.plainTextEdit_videoInfo.setPlainText(video_info_text)

            ret, frame = self.cap.read()
            frame = center_square(frame)
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

        server_uri1 = "ws://10.28.224.115:30057/ws"
        server_uri2 = "ws://10.28.224.115:30057/ws"

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

            # visualize figure 계산에 따른 위험 객체 판별 
            stat = ''
            if class_id != 9:
                # 각도 근사시 중앙 쏠림을 막기위한 보정치
                corr = 10

                # 객체 x축의 중간값을 각도로 수정 (-45 ~ 45도)
                r = ((x1 + x2) / 2 - 320) * (45 / 320)
                r = math.radians(r)

                # median 값을 실제 거리로 근사
                median_depth = 21 - (median_point * 4 / 3) + corr
                
                # depth와 각도에 따른 x, y값
                x = median_depth * math.sin(r)
                y = median_depth * math.cos(r) - corr

                # x축 너비의 따른 원 크기 조절
                rad = (x2 - x1) / 160

                if (x ** 2 + y ** 2) ** 0.5 - rad < 5: # 가까운 경우
                    stat = 'Danger'

                elif (x ** 2 + y ** 2) ** 0.5 - rad < 10: # 덜 가까운 경우
                    stat = 'Warning'

                else:
                    stat = 'Safe'
                    
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
            # text = f"{int(max_point)}, {int(middle_point)}"
            cv2.putText(frame, stat, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    async def send_frame(self, websocket1, websocket2, cap):
        frame_count = 0
        start_time = time.time()
        while cap.isOpened():
            current_time = time.time() - start_time
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            #print(f'current_time: {current_time}')
            #print(f'video_time: {video_time}')
            ret, frame = cap.read()
            frame = center_square(frame)
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

    def update_figure(self, metadata):
        self.mpl_canvas.clear_axes()
        for bbox in metadata:
            x1, y1, x2, y2 = bbox[:4]
            class_id = bbox[4]
            median_point = bbox[5]
            max_point = bbox[6]                     
            mean_point = bbox[7] 
            middle_point = bbox[8]
            if class_id != 9:
                # 각도 근사시 중앙 쏠림을 막기위한 보정치
                corr = 10

                # 객체 x축의 중간값을 각도로 수정 (-45 ~ 45도)
                r = ((x1 + x2) / 2 - 320) * (45 / 320)
                r = math.radians(r)

                # median 값을 실제 거리로 근사
                median_depth = 21 - (median_point * 4 / 3) + corr
                
                # depth와 각도에 따른 x, y값
                x = median_depth * math.sin(r)
                y = median_depth * math.cos(r) - corr

                # x축 너비의 따른 원 크기 조절
                rad = (x2 - x1) / 160

                if (x ** 2 + y ** 2) ** 0.5 - rad < 5: # 가까운 경우
                    if class_id == 0: # 사람인 경우
                        circle = Circle(xy=(x, y), radius=rad, edgecolor='red', facecolor='red')
                    else:
                        circle = Circle(xy=(x, y), radius=rad, edgecolor='blue', facecolor='blue')

                elif (x ** 2 + y ** 2) ** 0.5 - rad < 10: # 덜 가까운 경우
                    if class_id == 0: # 사람인 경우
                        circle = Circle(xy=(x, y), radius=rad, edgecolor='red', facecolor=(0.5, 0, 0))
                    else:
                        circle = Circle(xy=(x, y), radius=rad, edgecolor='blue', facecolor=(0, 0, 0.5))

                else:
                    if class_id == 0: # 사람인 경우
                        circle = Circle(xy=(x, y), radius=rad, edgecolor='red', facecolor='none')
                    else:
                        circle = Circle(xy=(x, y), radius=rad, edgecolor='blue', facecolor='none')

                self.mpl_canvas.axes.add_patch(circle)
            
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