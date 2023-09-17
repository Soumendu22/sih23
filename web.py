from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

outputFrame = None
lock = threading.Lock()
app = Flask(__name__)

vs = cv2.VideoCapture(0)
time.sleep(2.0)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

@app.route("/")
def index():
	return render_template("index.html")

def detect_motion(frameCount):
	global vs, outputFrame, lock
	total = 0
	while True:
		ret, frame = vs.read()
		frame = cv2.resize(frame, (640, 480))
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
		boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
		for (xA, yA, xB, yB) in boxes:
		    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
		    cv2.putText(frame, "Human!", (xA, yA), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2, cv2.LINE_AA)
		(flag, encodedImage) = cv2.imencode(".jpg", frame)
		with lock:
			outputFrame = frame.copy()

def generate():
	global outputFrame, lock
	while True:
		with lock:
			if outputFrame is None:
				continue
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			if not flag:
				continue
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")


app = Flask(__name__)

@app.route("/")
def video_feed():
	return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

vs.stop()
