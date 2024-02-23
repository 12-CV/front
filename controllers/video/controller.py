import cv2
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtGui import QImage
import threading
import queue
import websockets
import struct,  asyncio
import pickle
from common.const import *

class VideoSendThread(threading.Thread):
    def __init__(self, cap, frame_dict, frame_dict_lock):
        super().__init__()
        self.frame_dict = frame_dict
        self.frame_dict_lock = frame_dict_lock
        self.cap = cap
    async def send_frame(self):
        async with websockets.connect(SEND_URI) as websocket:
            frame_number = 0
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640,480))
                with self.frame_dict_lock:
                    self.frame_dict[f'{frame_number}'] = frame
                # 프레임 번호 전송
                await websocket.send(struct.pack('I', frame_number))
                # 프레임 전송
                buffer = cv2.imencode('.jpg', frame)[1].tobytes()
                await websocket.send(buffer)
                print(f"프레임 {frame_number} 전송 완료")
                frame_number += 1
                QThread.msleep(30)

    def run(self):
        print("VideoSendthread Started")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.send_frame())
        # asyncio.get_event_loop().run_forever()
        print("VideoSendthread Ended")


class VideoRecieveThread(threading.Thread):
    def __init__(self, frame_metadata_queue):
        super().__init__()
        self.frame_metadata_queue = frame_metadata_queue
    
    async def recieve_frame(self):
        async with websockets.connect(RECIEVE_URI) as websocket:
            while True:
                frame_number_packed = await websocket.recv()
                frame_metadata_packed = await websocket.recv()
                # number는 Integer, metadata는 Json
                frame_number = struct.unpack("I", frame_number_packed)[0]
                frame_metadata = pickle.loads(frame_metadata_packed)
                print(f"프레임 {frame_number} 수신 완료")
                print(f"프레임메타데이터 {frame_metadata} 수신 완료")
                self.frame_metadata_queue.put((frame_number, frame_metadata))

    def run(self):
        print("VideoRecieveThread Started")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.recieve_frame())
        # asyncio.get_event_loop().run_forever()
        print("VideoRecieveThread Ended")


class VideoRenderThread(QThread):
    video_signal = pyqtSignal(QImage)
    def __init__(self, frame_dict, frame_dict_lock, frame_metadata_queue):
        super().__init__(None)
        self.frame_dict = frame_dict
        self.frame_dict_lock = frame_dict_lock
        self.frame_metadata_queue = frame_metadata_queue
        
    def run(self):
        print("VideoRenderThread Started")
        while True:
            try:
                frame_number, frame_metadata = self.frame_metadata_queue.get()
            except queue.Empty:
                continue
            
            print(f"렌더러 프레임 {frame_number} 확인")
            with self.frame_dict_lock:
                frame = self.frame_dict[f'{frame_number}']
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_signal.emit(qt_image)