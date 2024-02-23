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
    def __init__(self, cap, frame_dict, frame_dict_lock, finish_event):
        super().__init__()
        self.frame_dict = frame_dict
        self.frame_dict_lock = frame_dict_lock
        self.cap = cap
        self.finish_event = finish_event
    async def send_frame(self):
        async with websockets.connect(SEND_URI) as websocket:
            frame_number = 0
            while self.cap.isOpened():
                ret, frame = self.cap.read()

                if self.finish_event.is_set():
                    break
                    
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
                frame_number += 1
                QThread.msleep(30)

    def run(self):
        print("VideoSendthread Started")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.send_frame())
        loop.close()
        self.finish_event.set()
        print("VideoSendthread Ended")


class VideoRecieveThread(threading.Thread):
    def __init__(self, frame_metadata_queue, finish_event):
        super().__init__()
        self.frame_metadata_queue = frame_metadata_queue
        self.finish_event = finish_event
    async def recieve_frame(self):
        async with websockets.connect(RECIEVE_URI) as websocket:
            while True:
                try:
                    frame_number_packed = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    frame_metadata_packed = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                except asyncio.TimeoutError:
                    if self.finish_event.is_set():
                        self.loop.stop()
                        break
                    continue
                # number는 Integer, metadata는 Json
                frame_number = struct.unpack("I", frame_number_packed)[0]
                frame_metadata = pickle.loads(frame_metadata_packed)
                # print(f"프레임 {frame_number} 수신 완료")
                # print(f"프레임메타데이터 {frame_metadata} 수신 완료")
                self.frame_metadata_queue.put((frame_number, frame_metadata))

    def run(self):
        print("VideoRecieveThread Started")
        loop = asyncio.new_event_loop()
        self.loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.recieve_frame())
        except:
            loop.close()
        self.finish_event.set()
        print("VideoRecieveThread Ended")


class VideoRenderThread(QThread):
    video_signal = pyqtSignal(QImage)
    interrupt_signal = False
    def __init__(self, frame_dict, frame_dict_lock, frame_metadata_queue: queue.Queue, finish_event):
        super().__init__(None)
        self.frame_dict = frame_dict
        self.frame_dict_lock = frame_dict_lock
        self.frame_metadata_queue = frame_metadata_queue
        self.finish_event = finish_event
        
    def run(self):
        print("VideoRenderThread Started")
        while True:
            try:
                frame_number, frame_metadata = self.frame_metadata_queue.get(timeout=0.1)
            except queue.Empty:
                if self.finish_event.is_set():
                    break
                continue
            
            print(f"렌더러 프레임 {frame_number} 확인")
            with self.frame_dict_lock:
                frame = self.frame_dict[f'{frame_number}']
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape

            for item in frame_metadata:
                predictions = item['predictions']
                for prediction in predictions:
                    label = prediction['label']
                    bbox = prediction['bbox']
                    confidence = prediction['confidence']

                    # bbox 그리기
                    cv2.rectangle(rgb_image, bbox['pt1'], bbox['pt2'],(255, 0, 0), thickness=2)

                    # label과 confidence 표시
                    text_position = (bbox['pt1'][0], bbox['pt1'][1] - 10)
                    text = f"{label}: {confidence}"
                    cv2.putText(rgb_image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_signal.emit(qt_image)
            QThread.msleep(20)
        print("VideoRenderThread Ended")