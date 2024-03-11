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

# 프로젝트 내의 모듈
from common.custom_class import CMainWindow
from common.function import *


class MainApp(CMainWindow):
    frame_queue = asyncio.Queue(3)
    metadata_queue = asyncio.Queue()
    task_list = []

    def __init__(self, mode):
        super().__init__()
        loadUi("./ui/main_window.ui", self)

        self.mode = mode
        self.pushButton_importVideo.clicked.connect(self.import_video_button_clicked)
        self.pushButton_startVideo.clicked.connect(self.start_video_button_clicked)
        self.pushButton_stopVideo.clicked.connect(self.stop_video_button_clicked)
        self.pushButton_restartVideo.clicked.connect(self.restart_video_button_clicked)

    def import_video_button_clicked(self):
        # file_name, _ = QFileDialog.getOpenFileName(self, "영상 불러오기", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        file_name, _ = QFileDialog.getOpenFileName(self, "영상 불러오기", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.file_name = file_name

            self.cap = cv2.VideoCapture(self.file_name)
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            # self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.video_width = 640
            self.video_height = 640
            video_info_text = f"\n파일 이름 : \n{self.file_name} \n\nfps : {self.fps} \n\nvideo size : {self.video_width} x {self.video_height}"
            self.plainTextEdit_videoInfo.setPlainText(video_info_text)

            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640,640))
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

        server_uri = "ws://10.28.224.34:30348/ws"
        async with websockets.connect(server_uri) as websocket:
            # await asyncio.gather(
            #     self.send_frame(websocket, self.cap), 
            #     self.receive_frame(websocket), 
            #     self.render_frame((self.video_height, self.video_width), self.fps)
            # )

             # 각 작업을 Task로 생성
            self.send_task = asyncio.create_task(self.send_frame(websocket, self.cap))
            self.receive_task = asyncio.create_task(self.receive_frame(websocket))
            self.render_task = asyncio.create_task(self.render_frame((self.video_height, self.video_width), self.fps))

            self.task_list.append(self.send_task)
            self.task_list.append(self.receive_task)
            self.task_list.append(self.render_task)
            # 생성된 모든 Task들을 기다립니다.
            await asyncio.gather(
                self.send_task, 
                self.receive_task, 
                self.render_task,
                return_exceptions=True  # 각 태스크의 예외를 반환하도록 설정
            )


    def update_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label_videoWindow.setPixmap(QPixmap.fromImage(qt_image))
    
    def draw_bboxes(self, frame, bboxes, frame_shape):
        for bbox in bboxes:
            x, y, w, h = bbox
            # 원래 프레임의 크기에 맞게 bbox 좌표 조정
            x = int(x * frame_shape[1] / 640)
            y = int(y * frame_shape[0] / 640)
            w = int(w * frame_shape[1] / 640)
            h = int(h * frame_shape[0] / 640)
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

    async def send_frame(self, websocket, cap):
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == None:
                break
            try:
                frame = cv2.resize(frame, (640, 640))
            except:
                raise Exception("영상이 끝났습니다!")
            
            await self.frame_queue.put(frame)
            _, img_encoded = cv2.imencode('.jpg', frame)
            await websocket.send(img_encoded.tobytes())

    async def receive_frame(self, websocket):
        while True:
            result = await websocket.recv()
            result_dict = json.loads(result)
            bboxes = result_dict.get("bboxes", [])
            await self.metadata_queue.put(bboxes)

    async def render_frame(self, frame_shape, fps):
        time_per_frame = 1 / fps
        prev_time = time.time() - 1
        while True:
            metadata = await self.metadata_queue.get()
            frame = await self.frame_queue.get()
            self.draw_bboxes(frame, metadata, frame_shape)

            curr_time = time.time()
            elapsed = curr_time - prev_time
            if elapsed < time_per_frame:
                await asyncio.sleep(time_per_frame - elapsed)

            # cv2.imshow("Object Detection", frame)
            self.update_frame(frame)
            prev_time = time.time()
            
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