from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import cvzone
from ultralytics import YOLO
import math
import numpy
from sort import *
from datetime import datetime
import torch

torch.cuda.set_device(0)

open('logs.txt', 'w').close()

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

# vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0, resolution=(640, 480)).start()
# vs.stream.set(3, 1280)
# vs.stream.set(4, 720)
time.sleep(2.0)

tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)
totalCount = []
activeCount = []

model = YOLO("../Yolo-Weights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/main.html")
def main_page():
    return render_template("main.html")


def detect_motion(frameCount):
    global vs, outputFrame, lock, currentClass, id, position, logs, conf
    while True:
        img = vs.read()
        # imgRegion = cv2.bitwise_and(img, mask)
        results = model(img, stream=True)
        detections = np.empty((0, 5))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Confidence
                conf = math.ceil((box.conf[0] * 100))
                print(conf)
                # Class Names
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                if ((currentClass == "person")
                        and conf > 45):
                    currentArray = numpy.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
        resultsTracker = tracker.update(detections)
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            print(result)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=8, t=2, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(0, y1 + 35)), scale=2,
                               thickness=2, offset=7)
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            # if cx < 320 and cy < 240:  # Telling the General Location of the Detection
            #     position = "Top Left"
            # elif cx > 320 and cy > 240:
            #     position = "Bottom Right"
            # elif cx > 320 and cy < 240:
            #     position = "Top Right"
            # elif cx < 320 and cy > 240:
            #     position = "Bottom Left"
            # else:
            #     position = "Centre"

            if 0 < cx < 640 and 0 < cy < 480:
                if activeCount.count(id) == 0:
                    activeCount.append(id)
                for a in activeCount:
                    if a not in totalCount:
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        print(f'{current_time} : Person Spotted')  # Log Printing
                        with open('logs.txt', 'a') as logs:
                            logs.write(f'\n{current_time} : Person Spotted')
                if totalCount.count(id) == 0:
                    totalCount.append(id)
        cvzone.putTextRect(img, f'TD :{len(totalCount)}  AD :{len(activeCount)}', (0, 25), offset=5, scale=2)
        # cvzone.putTextRect(img, f'AD :{len(activeCount)}', (0, 100))
        with lock:
            outputFrame = img.copy()
        activeCount.clear()


def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/data')
def data():
    with open('logs.txt') as f:
        yield f.read()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

vs.stop()
logs.close()
