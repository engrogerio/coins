import numpy as np
import cv2
from picamera2 import Picamera2
from libcamera import controls
import time
import os


def pre_process(img):
	img_pre = cv2.GaussianBlur(img, (5, 5), 3)
	img_pre = cv2.Canny(img_pre, 90, 140)
	kernel = np.ones((4, 4), np.uint8)
	img_pre = cv2.dilate(img_pre, kernel, iterations=2)
	img_pre = cv2.erode(img_pre, kernel, iterations=1)
	return img_pre

def save_img(img, x, y, w, h):
	value = '10cents'
	BASEDIR = os.path.dirname(os.path.realpath(__file__))
	folder = os.path.join(BASEDIR, value)
	_, _, files = next(os.walk(folder))
	file_count = len(files)
	filename = os.path.join(folder, f'{value}{file_count}.jpg')
	crop = img[y:y + h, x:x + w]
	crop = cv2.resize(crop, (224,224))
	cv2.imwrite(filename, crop)

picam2 = Picamera2()
picam2.start(show_preview=False)

picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "AfSpeed": controls.AfSpeedEnum.Fast})

#picam2.start_and_capture_files("fastfocus-test{:d}.jpg", num_files=3, delay=0.5)

while True:
	# Read a frame from the camera
	bgr_img = np.array(picam2.capture_image('main'))

	# back to RGB
	img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
	img_processed = pre_process(img)
	contours, hi = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area > 2000:
			x, y, w, h = cv2.boundingRect(cnt)
			if cv2.waitKey(1) & 0xFF == ord('p'):
				save_img(img, x, y, w, h)
			cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
	# Display the frame
	cv2.imshow('Live Camera Feed', img)
	cv2.imshow('Processed image', img_processed)
	
	# Exit the loop if the 'q' key is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release the camera and close the OpenCV window
picam2.release()
cv2.destroyAllWindows()
