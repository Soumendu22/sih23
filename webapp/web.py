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
# import torch

# torch.cuda.set_device(0)

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]
totalcount=0

model = YOLO("yolov8n.pt")
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
	global vs, outputFrame, lock, totalcount
	while True:
		# img = vs.read()
		# results = model(img, classes=0, stream=True)
		# for r in results:
		# 	boxes = r.boxes
		# 	for box in boxes:
		# 		x1, y1, x2,  y2 = box.xyxy[0]
		# 		x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
		# 		w, h = (x2-x1), (y2-y1)
		# 		print(x1, y1, x2, y2)
		# 		cvzone.cornerRect(img, (x1, y1, w, h))
		# 		conf = math.ceil((box.conf[0]*100))
		# 		print(conf)
		# 		cls = int(box.cls[0])
		# 		cvzone.putTextRect(img, f'{classNames[cls]} {conf}%', (max(0, x1), max(0, y1 + 35)), scale=1, thickness=1)	
		
		img = vs.read()
		# imgRegion = cv2.bitwise_and(img, mask)
		results = model(img, stream=True)
		detections = np.empty((0, 5))
		for r in results:
		    boxes = r.boxes
		    for box in boxes:
		        # Bounding Box
		        x1, y1, x2, y2 = box.xyxy[0]
		        # x1, y1, w, h = box.xywh[0]
		        # x1, y1, w, h = int(x1), int(y1), int(w), int(h)
		        # cv2.rectangle(img,(x1, y1), (x2, y2), (255,0,255),3)
		        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
		        w, h = (x2 - x1), (y2 - y1)
		        print(x1, y1, x2, y2)
		        # cvzone.cornerRect(img, (x1, y1, w, h), l=8, t=2)
		        # Confidence
		        conf = math.ceil((box.conf[0] * 100))
		        print(conf)
		        # Class Names
		        cls = int(box.cls[0])
		        currentClass = classNames[cls]
		        if ((currentClass == "person")
		                and conf > 45):
		            # cvzone.cornerRect(img, (x1, y1, w, h), l=8, t=2, rt=5)
		            # cvzone.putTextRect(img, f'{classNames[cls]} {conf}%', (max(0, x1), max(0, y1 + 35)), scale=0.75,
		            #                    thickness=1, colorR=(0, 0, 0), colorT=(255, 255, 255), offset=5)
		            currentArray = numpy.array([x1, y1, x2, y2, conf])
		            detections = np.vstack((detections, currentArray))	
		resultsTracker = tracker.update(detections)
		cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), 5)
		for result in resultsTracker:
		    x1, y1, x2, y2, id = result
		    print(result)
		    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
		    w, h = x2-x1, y2-y1
		    cvzone.cornerRect(img, (x1, y1, w, h), l=8, t=2, rt=2, colorR=(255, 0, 255))
		    cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(0, y1 + 35)), scale=2,
		                       thickness=2,  offset=7)
		    cx, cy = x1+w//2, y1+h//2
		    cv2.circle(img, (cx, cy),5,(255, 0, 255), cv2.FILLED)	
		    if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[1]+20:
		        totalcount += 1	
		cvzone.putTextRect(img, f'Count :{totalcount}', (50, 50))	
		with lock:
			outputFrame = img.copy()

def generate():
	global outputFrame, lock
	while True:
		with lock:
			if outputFrame is None:
				continue
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			if not flag:
				continue
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

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
