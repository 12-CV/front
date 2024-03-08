import cv2
import numpy as np
import io
from time import time
import asyncio
import websockets
import json
import struct

def draw_bboxes(frame, bboxes, frame_shape):
    for bbox in bboxes:
        x, y, w, h = bbox
        # 원래 프레임의 크기에 맞게 bbox 좌표 조정
        x = int(x * frame_shape[1] / 640)
        y = int(y * frame_shape[0] / 640)
        w = int(w * frame_shape[1] / 640)
        h = int(h * frame_shape[0] / 640)
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

async def send_frame(websocket, frame):
    frame = cv2.resize(frame, (640, 640))
    _, img_encoded = cv2.imencode('.jpg', frame)
    await websocket.send(img_encoded.tobytes())
    #print(f"프레임 전송 완료")

async def receive_detection_results(websocket, frame_shape, frame):
    result = await websocket.recv()
    #print(f"프레임메타데이터 수신 완료")
    result_dict = json.loads(result)
    bboxes = result_dict.get("bboxes", [])
    draw_bboxes(frame, bboxes, frame_shape)

async def main():
    video_path = "walk.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    server_uri = "ws://10.28.224.34:30348/ws"
    async with websockets.connect(server_uri) as websocket:
        while cap.isOpened():
            ret, frame = cap.read(15)
            if not ret:
                break

            start = time()
            await send_frame(websocket, frame)
            #print(f"Send Frame Time: {time() - start}")

            start = time()
            await receive_detection_results(websocket, (height, width), frame)
            #print(f"Receive Results Time: {time() - start}")

            start = time()
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #print(f"Show Video Time: {time() - start}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())