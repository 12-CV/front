import cv2
import numpy as np
import io
from time import time
import asyncio
import websockets
import json
import struct

frame_queue = asyncio.Queue(30)
metadata_queue = asyncio.Queue()

def draw_bboxes(frame, bboxes, frame_shape):
    for bbox in bboxes:
        x, y, w, h = bbox
        # 원래 프레임의 크기에 맞게 bbox 좌표 조정
        x = int(x * frame_shape[1] / 640)
        y = int(y * frame_shape[0] / 640)
        w = int(w * frame_shape[1] / 640)
        h = int(h * frame_shape[0] / 640)
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

async def send_frame(websocket, cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == None:
            break

        await frame_queue.put(frame)
        try:
            frame = cv2.resize(frame, (640, 640))
        except:
            raise Exception("영상이 끝났습니다!")
        _, img_encoded = cv2.imencode('.jpg', frame)
        await websocket.send(img_encoded.tobytes())

async def receive_frame(websocket):
    while True:
        result = await websocket.recv()
        result_dict = json.loads(result)
        bboxes = result_dict.get("bboxes", [])
        await metadata_queue.put(bboxes)

async def render_frame(frame_shape, fps):
    time_per_frame = 1 / fps
    prev_time = time() - 1
    while True:
        metadata = await metadata_queue.get()
        frame = await frame_queue.get()
        draw_bboxes(frame, metadata, frame_shape)

        curr_time = time()
        elapsed = curr_time - prev_time
        if elapsed < time_per_frame:
            await asyncio.sleep(time_per_frame - elapsed)

        cv2.imshow("Object Detection", frame)
        prev_time = time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise Exception("키보드로 인한 종료!")

async def main():
    video_path = "walk.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    server_uri = "ws://10.28.224.34:30348/ws"
    async with websockets.connect(server_uri) as websocket:
        await asyncio.gather(send_frame(websocket, cap), receive_frame(websocket), render_frame((height, width), fps))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())