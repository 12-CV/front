# 표준 라이브러리
import argparse
import asyncio
import json
import sys
import time
import typing

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

server_uris = [
    # "ws://10.28.224.34:30348/ws", # 재영서버
    "ws://10.28.224.34:30349/ws", # 재영 리뉴얼 서버
    # "ws://10.28.224.216:30394/ws", # 혜나서버
    # "ws://10.28.224.181:30316/ws", # 민주서버
    # "ws://10.28.224.115:30057/ws", # 동우서버
    # "ws://10.28.224.52:30300/ws", # 세진서버
    # "ws://10.28.224.52:30301/ws", # 세진 리뉴얼 서버
]
FRAME_QUEUE_LIMIT = 1

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
    def __init__(self, parent=None, width=4, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.set_facecolor('black')
        self.axes.tick_params(axis='both', colors='white', labelcolor='white')
        self.clear_axes()

        super(MyMplCanvas, self).__init__(fig)
        self.setFixedSize(640, 640)

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
    frame_queue = asyncio.Queue(FRAME_QUEUE_LIMIT)
    metadata_queue = asyncio.Queue()
    temp_queue = asyncio.Queue()
    task_list = []

    def __init__(self, mode):
        super().__init__()
        loadUi("./ui/main_window.ui", self)
        
        self.mpl_canvas = MyMplCanvas(self)
        self.horizontalLayout_videoLayout.addWidget(self.mpl_canvas)
        self.last_beep_time = time.time()
        self.beep_player = QMediaPlayer()
        
        self.mode = mode
        self.pushButton_importVideo.clicked.connect(self.import_video_button_clicked)
        self.pushButton_importCamera.clicked.connect(self.import_camera_button_clicked)
        self.pushButton_startVideo.clicked.connect(self.start_video_button_clicked)
        self.pushButton_stopVideo.clicked.connect(self.stop_video_button_clicked)
        self.pushButton_restartVideo.clicked.connect(self.restart_video_button_clicked)
        
    async def custom_init(self):
        self.connections = await self.connect_servers(server_uris)
        global FRAME_QUEUE_LIMIT
        FRAME_QUEUE_LIMIT = len(self.connections) * 2
        print(f"연결된 서버의 개수는 {len(self.connections)} 입니다.")

    def play_beep(self, y):
        current_time = time.time()
        interval = 1.5  # 기본 간격
        interval = 999999  # 기본 간격

        if y < 5:
            interval = 0.1
        # elif y < 10:
        #     interval = 0.6

        if current_time - self.last_beep_time >= interval:
            # 오디오 파일 경로 설정
            audioFile = 'beep.mp3'
            url = QUrl.fromLocalFile(audioFile)
            content = QMediaContent(url)
            
            self.beep_player.setMedia(content)
            self.beep_player.play()
            self.last_beep_time = current_time  # 마지막 소리 재생 시간 업데이트

    @qasync.asyncSlot()
    async def import_camera_button_clicked(self):
        await self.stop_video_button_clicked()
        self.frame_queue = asyncio.Queue(FRAME_QUEUE_LIMIT)
        self.metadata_queue = asyncio.Queue()
        self.cap = cv2.VideoCapture(0)
        self.file_name = 'Camera'
        if self.cap:
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
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
        self.frame_queue = asyncio.Queue(FRAME_QUEUE_LIMIT)
        self.metadata_queue = asyncio.Queue()
        file_name, _ = QFileDialog.getOpenFileName(self, "영상 불러오기", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.file_name = file_name

            self.cap = cv2.VideoCapture(self.file_name)
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
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
        self.frame_queue = asyncio.Queue(FRAME_QUEUE_LIMIT)
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
        
        # self.task_list.append(asyncio.create_task(self.send_frame(self.connections, self.cap)))
        # self.task_list.extend([asyncio.create_task(self.receive_frame(socket)) for socket in self.connections])
        # self.task_list.append(asyncio.create_task(self.render_frame((self.video_height, self.video_width), self.fps)))

        self.task_list.append(asyncio.create_task(self.send_frame(self.connections, self.cap)))
        self.task_list.extend([asyncio.create_task(self.receive_frame2(socket)) for socket in self.connections])
        self.task_list.append(asyncio.create_task(self.render_frame2((self.video_height, self.video_width), self.fps)))
        
        await asyncio.gather(
            *self.task_list,
            # return_exceptions=True  
        )

    async def connect_servers(self, server_uris):
        async def check_websocket(uri):
            try:
                socket = await websockets.connect(uri, timeout=1)
                return socket
            except Exception as e:
                return None
        
        tasks = [check_websocket(uri) for uri in server_uris]
        results = await asyncio.gather(*tasks)
        
        valid_connections = [result for result in results if result is not None]

        return valid_connections

    def update_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label_videoWindow.setPixmap(QPixmap.fromImage(qt_image))
    
    async def send_frame(self, sockets, cap):
        print(f"프레임 보내기를 시작합니다.")
        print(f"받은 서버목록 {sockets}")

        frame_count = 0
        start_time = time.time()
        while cap.isOpened():

            current_time = time.time() - start_time
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            ret, frame = cap.read()
            frame = center_square(frame)
            # if (current_time - video_time) > 1:
            #     continue
            if ret == None:
                break
            try:
                frame = cv2.resize(frame, (self.video_width, self.video_height))
            except:
                raise Exception("영상이 끝났습니다!")
            
            await self.frame_queue.put(frame)
            _, img_encoded = cv2.imencode('.jpg', frame)

            await sockets[frame_count % len(sockets)].send(img_encoded.tobytes())
            frame_count += 1

    async def receive_frame(self, websocket):
        while True:
            result = await websocket.recv()
            result_dict = json.loads(result)
            bboxes = result_dict.get("bboxes", [])
            await self.metadata_queue.put(bboxes)

    async def receive_frame2(self, websocket):
        while True:
            data = await websocket.recv()
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            await self.temp_queue.put(image)

    async def render_frame(self, frame_shape, fps):
        FPS = fps
        time_per_frame = 1 / fps
        prev_time = time.time() - 1
        fps_time = time.time()
        frame_count = 0
        while True:
            metadata = await self.metadata_queue.get()
            frame = await self.frame_queue.get()

            curr_time = time.time()
            elapsed = curr_time - prev_time

            if 1 < (time.time() - fps_time):
                FPS = frame_count
                frame_count = 0
                fps_time = time.time()

            cv2.putText(frame, f"FPS: {FPS}", (frame.shape[1] - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            if elapsed < 0.03:
                await asyncio.sleep(time_per_frame - elapsed)

            self.update_figure(metadata, frame)
            self.update_frame(frame)
            prev_time = time.time()

            frame_count += 1


    async def render_frame2(self, frame_shape, fps):
        FPS = fps
        time_per_frame = 1 / fps
        prev_time = time.time() - 1
        fps_time = time.time()
        frame_count = 0
        while True:
            frame = await self.temp_queue.get()
            frame = await self.frame_queue.get()
            curr_time = time.time()
            elapsed = curr_time - prev_time

            if 1 < (time.time() - fps_time):
                FPS = frame_count
                frame_count = 0
                fps_time = time.time()

            cv2.putText(frame, f"FPS: {FPS}", (frame.shape[1] - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            if elapsed < 0.03:
                await asyncio.sleep(time_per_frame - elapsed)

            self.update_frame(frame)
            prev_time = time.time()

            frame_count += 1

    def update_figure(self, metadata, frame):
        self.mpl_canvas.clear_axes()
        for bbox in metadata:
            x1, y1, x2, y2 = bbox[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = bbox[4]
            median_point = bbox[5]
            max_point = bbox[6]                     
            mean_point = bbox[7] 
            middle_point = bbox[8]
            x = bbox[9]
            y = bbox[10]
            rad = bbox[11]
            distance = bbox[12]

            if class_id != 9:
                if distance < 5:
                    stat = 'Danger'
                    color = (0, 0, 255)
                    color_str = "red"

                elif distance < 10:
                    stat = 'Warning'
                    color = (0, 165, 255)
                    color_str = "orange"

                else:
                    stat = 'Safe'
                    color = (0, 255, 0)
                    color_str = "green"

                circle = Circle(xy=(x, y), radius=rad, edgecolor=color_str, facecolor=color_str)
                self.play_beep(distance)
                self.mpl_canvas.axes.add_patch(circle)

                # 거리 20 이내의 객체만 Bbox를 그려준다.
                if distance < 20:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, stat, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            else:
                mosaic_area = frame[y1:y2, x1:x2]
                X, Y = x1//30, y1//30
                if X <= 0:
                    X = 1
                if Y <= 0:
                    Y = 1
                mosaic_area = cv2.resize(mosaic_area, (X,Y))
                mosaic_area = cv2.resize(mosaic_area, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                frame[y1:y2, x1:x2] = mosaic_area
            
        self.mpl_canvas.draw()
def main():
    parser = argparse.ArgumentParser(description='Program Mode')
    parser.add_argument('--mode', type=str, help='DEV or PROD', default="DEV")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    main_window = MainApp(args.mode)
    loop.create_task(main_window.custom_init())
    main_window.show()

    with loop:
        loop.run_forever()
    for con in main_window.connections:
        con.close()
    
if __name__ == '__main__':
    main()