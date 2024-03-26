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
from matplotlib.patches import Ellipse, Circle

server_uris = [
    "ws://10.28.224.34:30349/ws", # 재영 리뉴얼 서버
    "ws://10.28.224.216:30395/ws", # 혜나 리뉴얼 서버
    "ws://10.28.224.181:30317/ws", # 민주 리뉴얼 서버
    "ws://10.28.224.115:30058/ws", # 동우 리뉴얼 서버
    "ws://10.28.224.52:30300/ws", # 세진 리뉴얼 서버
]

FRAME_QUEUE_LIMIT = 5 # 웹캠 -> 1, 영상 10~20


frame_counter = 0
lock = asyncio.Lock()
condition = asyncio.Condition()
time_count_dict = {}

class MainApp(CMainWindow):
    frame_queue = asyncio.Queue(FRAME_QUEUE_LIMIT)
    metadata_queue = asyncio.Queue()
    recieved_frame_queue = asyncio.Queue()
    recieved_radar_queue = asyncio.Queue()
    recieved_beep_queue = asyncio.Queue()
    task_list = []

    def __init__(self, mode):
        super().__init__()
        loadUi("./ui/main_window.ui", self)
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
        FRAME_QUEUE_LIMIT = len(self.connections) * 1
        print(f"연결된 서버의 개수는 {len(self.connections)} 입니다.")

    def play_beep(self, y=3):
        current_time = time.time()
        interval = 1.5  # 기본 간격
        interval = 999999  # 기본 간격

        if y < 5:
            interval = 0.2
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
            frame = self.center_square(frame)
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
            frame = self.center_square(frame)

            if ret:
                frame = cv2.resize(frame, (self.video_width, self.video_height))
                self.label_videoWindow.setPixmap(frame_to_pixmap(frame))            
        else:
            show_message(self, "파일이 존재하지 않습니다.")
            return
        
    @qasync.asyncSlot()
    async def restart_video_button_clicked(self):
        await self.stop_video_button_clicked()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        await self.start_video_button_clicked()
    
    @qasync.asyncSlot()
    async def stop_video_button_clicked(self):
        for task in self.task_list:
            task.cancel()
        for socket in self.connections:
            await socket.close()
        self.connections.clear()
        self.frame_queue = asyncio.Queue(FRAME_QUEUE_LIMIT)
        self.metadata_queue = asyncio.Queue()

    @qasync.asyncSlot()
    async def start_video_button_clicked(self):
        file_name = getattr(self, 'file_name', None)

        if not file_name:
            show_message(self, "파일을 먼저 선택해주세요!")
            return
        self.task_list = []
        self.connections = await self.connect_servers(server_uris)
        global frame_counter
        frame_counter = 0
        self.task_list.append(asyncio.create_task(self.send_frame(self.connections, self.cap)))
        self.task_list.extend([asyncio.create_task(self.receive_frame(socket, len(self.connections))) for socket in self.connections])
        self.task_list.append(asyncio.create_task(self.render_frame((self.video_height, self.video_width), self.fps)))
        
        await asyncio.gather(
            *self.task_list,
            # return_exceptions=True  
        )

    def center_square(self, image):
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
    
    def update_radar(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label_radarWindow.setPixmap(QPixmap.fromImage(qt_image))
    
    async def send_frame(self, sockets, cap):
        print(f"프레임 보내기를 시작합니다.")
        print(f"받은 서버목록 {sockets}")

        frame_count = 0
        start_time = time.time()
        blur_face = self.checkBox_mosaicFace.isChecked()
        draw_bbox = self.checkBox_objectDetect.isChecked()
        collision_warning = self.checkBox_collisionDetect.isChecked()

        while cap.isOpened():
            current_time = time.time() - start_time
            time_count_dict[frame_count] = time.time()

            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            ret, frame = cap.read()
            frame = self.center_square(frame)
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
            
            metadata = {
                "frame_count":frame_count,
                "blur_face":blur_face,
                "draw_bbox":draw_bbox,
                "collision_warning":collision_warning,
            }
            await sockets[frame_count % len(sockets)].send(img_encoded.tobytes())
            await sockets[frame_count % len(sockets)].send(json.dumps(metadata))
            frame_count += 1

    async def receive_frame(self, websocket, sockets):
        global frame_counter

        while True:
            frame_data = await websocket.recv()
            radar_data = await websocket.recv()
            metadata_json = await websocket.recv()
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            radar = cv2.imdecode(np.frombuffer(radar_data, dtype=np.uint8), cv2.IMREAD_COLOR)

            metadata = json.loads(metadata_json)
            frame_count = metadata["frame_count"]
            while True:
                async with lock:
                    if frame_counter == frame_count:
                        break
                async with condition:
                    await condition.wait()
            
            current_time = time.time()
            server_num = f"연결된 서버 개수: {sockets}\n\n"
            time_info_text = f"딜레이시간: {(current_time - time_count_dict[frame_counter]):.3f}"
            self.plainTextEdit_mainLogger.setPlainText(server_num + time_info_text)
            async with lock:
                frame_counter += 1

            async with condition:
                condition.notify_all()
            
            await self.recieved_frame_queue.put(frame)
            await self.recieved_radar_queue.put(radar)
            await self.recieved_beep_queue.put(metadata["beep"])

    async def render_frame(self, frame_shape, fps):
        FPS = fps
        time_per_frame = 0.031
        prev_time = time.time() - 1
        fps_time = time.time()
        frame_count = 0
        while True:
            frame = await self.frame_queue.get()
            frame = await self.recieved_frame_queue.get()
            radar = await self.recieved_radar_queue.get()
            beep = await self.recieved_beep_queue.get()
            curr_time = time.time()
            elapsed = curr_time - prev_time

            if 1 < (time.time() - fps_time):
                FPS = frame_count
                frame_count = 0
                fps_time = time.time()

            cv2.putText(frame, f"FPS: {FPS}", (frame.shape[1] - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.putText(frame, f"FPS: {30}", (frame.shape[1] - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            if elapsed < time_per_frame:
                await asyncio.sleep(time_per_frame - elapsed)

            self.update_frame(frame)
            self.update_radar(radar)
            prev_time = time.time()
            if beep:
                self.play_beep()

            frame_count += 1

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