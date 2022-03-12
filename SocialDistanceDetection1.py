# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:42:33 2021

@author: Rames
"""

# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# define the minimum safe distance (in pixels) that two people can be
# from each other
MIN_DISTANCE = 50

# import the necessary packages
import streamlit as st
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# Load Yolo
# Download weight file(yolov3_training_2000.weights) from this link :- https://drive.google.com/file/d/10uJEsUpQI3EmD98iwrwzbD4e19Ps-LHZ/view?usp=sharing
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

st.title('RBL Project')
st.sidebar.title('RBL Project Sidebar')
FRAME_WINDOW = st.image([])

def main():
	a = st.sidebar.selectbox("Select", ["Video","Video Summarization","Retrain Model", "Exit"])
	print('A value is',a)
	if a == 'Video':
		call_video()
	elif a== 'Video Summarization':
		video_summarization()
	elif a=='Exit':
		pass

def call_video():
	def detect_weapon(img):
		blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
		height, width, channels = img.shape
		net.setInput(blob)
		outs = net.forward(output_layers)

		# Showing information on the screen
		class_ids = []
		confidences = []
		boxes = []
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.5:
					# Object detected
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)
					# Rectangle coordinates
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)

					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					class_ids.append(class_id)

		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
		# print(indexes)
		if indexes == 0: print("weapon detected in frame")
		font = cv2.FONT_HERSHEY_PLAIN
		
		# frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
		cv2.imshow("Image", img)
		return boxes,indexes,font,class_ids

	def detect_people(frame, net, ln, personIdx=0):
		# grab the dimensions of the frame and  initialize the list of
		# results
		(H, W) = frame.shape[:2]
		results = []

		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		layerOutputs = net.forward(ln)

		# initialize our lists of detected bounding boxes, centroids, and
		# confidences, respectively
		boxes = []
		centroids = []
		confidences = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability)
				# of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter detections by (1) ensuring that the object
				# detected was a person and (2) that the minimum
				# confidence is met
				if classID == personIdx and confidence > MIN_CONF:
					# scale the bounding box coordinates back relative to
					# the size of the image, keeping in mind that YOLO
					# actually returns the center (x, y)-coordinates of
					# the bounding box followed by the boxes' width and
					# height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top
					# and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates,
					# centroids, and confidences
					boxes.append([x, y, int(width), int(height)])
					centroids.append((centerX, centerY))
					confidences.append(float(confidence))

		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)



		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# update our results list to consist of the person
				# prediction probability, bounding box coordinates,
				# and the centroid
				r = (confidences[i], (x, y, x + w, y + h), centroids[i])
				results.append(r)

		# return the list of results
		return results

	# USAGE
	# python social_distance_detector.py --input pedestrians.mp4
	# python social_distance_detector.py --input pedestrians.mp4 --output output.avi

	# import the necessary packages
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", type=str, default="",
		help="path to (optional) input video file")
	ap.add_argument("-o", "--output", type=str, default="",
		help="path to (optional) output video file")
	ap.add_argument("-d", "--display", type=int, default=1,
		help="whether or not output frame should be displayed")
	args = vars(ap.parse_args(["--input","","--output","my_output.avi","--display","1"]))

	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join(["coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join(["yolov3.weights"])
	configPath = os.path.sep.join(["yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

	# initialize the video stream and pointer to output video file
	print("[INFO] accessing video stream...")
	vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
	# vs = cv2.VideoCapture(0)
	writer = None

	# loop over the frames from the video stream
	while True:
		# read the next frame from the file
		(grabbed, frame) = vs.read()

		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break
		
		# resize the frame and then detect people (and only people) in it
		frame = imutils.resize(frame, width=700)
		results = detect_people(frame, net, ln,
			personIdx=LABELS.index("person"))
		boxes,indexes,font,class_ids = detect_weapon(frame)

		# initialize the set of indexes that violate the minimum social
		# distance
		violate = set()

		# ensure there are *at least* two people detections (required in
		# order to compute our pairwise distance maps)
		if len(results) >= 2:
			# extract all centroids from the results and compute the
			# Euclidean distances between all pairs of the centroids
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")

			# loop over the upper triangular of the distance matrix
			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):
					# check to see if the distance between any two
					# centroid pairs is less than the configured number
					# of pixels
					if D[i, j] < MIN_DISTANCE:
						# update our violation set with the indexes of
						# the centroid pairs
						violate.add(i)
						violate.add(j)
			for i in range(len(boxes)):
				if i in indexes:
					x,y,w,h = boxes[i]
					label = str(classes[class_ids[i]])
					color = colors[class_ids[i]]
					cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
					cv2.putText(frame,label,(x,y+30),font,3,color,3)
			

		# loop over the results
		for (i, (prob, bbox, centroid)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then
			# initialize the color of the annotation
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)

			# if the index pair exists within the violation set, then
			# update the color
			if i in violate:
				color = (0, 0, 255)

			# draw (1) a bounding box around the person and (2) the
			# centroid coordinates of the person,
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.circle(frame, (cX, cY), 5, color, 1)

		# draw the total number of social distancing violations on the
		# output frame
		text = "Social Distancing Violations: {}".format(len(violate))
		cv2.putText(frame, text, (10, frame.shape[0] - 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

		# check to see if the output frame should be displayed to our
		# screen
		if args["display"] > 0:
			# show the output frame
			# cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
			
		# if an output video file path has been supplied and the video
		# writer has not been initialized, do so now
		# if args["output"] != "" and writer is None:
		# 	# initialize our video writer
		# 	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		# 	writer = cv2.VideoWriter(args["output"], fourcc, 25,
		# 		(frame.shape[1], frame.shape[0]), True)

		# if the video writer is not None, write the frame to the output
		# video file
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		FRAME_WINDOW.image(frame)

def video_summarization():
	pass

if __name__ == "__main__":
    main()