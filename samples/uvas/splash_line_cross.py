"""
Mask R-CNN
Train on the toy Grape dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
	   the command line as such:

	# Train a new model starting from pre-trained COCO weights
	python3 grape.py train --dataset=/path/to/grape/dataset --weights=coco

	# Resume training a model that you had trained earlier
	python3 grape.py train --dataset=/path/to/grape/dataset --weights=last

	# Train a new model starting from ImageNet weights
	python3 grape.py train --dataset=/path/to/grape/dataset --weights=imagenet

	# Apply color splash to an image
	python3 grape.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

	# Apply color splash to video using the last weights you trained
	python3 grape.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import time

import pickle
import cv2
import numpy as np
import skimage.draw
from os import getenv
from IPython.core.display import display, HTML

from deep_sort_pytorch.deep_sort import build_tracker
from deep_sort_pytorch.utils.draw import draw_boxes
from deep_sort_pytorch.utils.parser import get_config

from utils.text_utils import draw_text_area

import os.path
from os import path
import argparse

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import model as modellib, utils
import mrcnn.visualize
from mrcnn.config import Config

# Path to trained weights file #CHECK:
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Setting some global variables
VIDEO_CAPTURE_WIDTH = 1280
VIDEO_CAPTURE_HEIGHT = 720

# Setting the line crossing
LINE_COEFF = 0.5
X_LINE = int(LINE_COEFF * VIDEO_CAPTURE_WIDTH)
VIDEO_DIRECTION = 'right' # or left. If the first grape bunches appear of the left part of the video
DISTANCE_PER_FRAME = 1 # suppose that the video was recorded with constant velocity
											 # then the distance per frame is setting accordingly.
											 # [cm/frames]

											 # TODO: set this value to a more accurate value. 
											 # If you have the hilera distance, you can get 
											 # DISTANCE_PER_FRAME = hilera distance / video duration

############################################################
#  Configurations
############################################################


class InferenceConfig(Config):
	"""Configuration for training on the toy  dataset.
	Derives from the base Config class and overrides some values.
	"""
	# Give the configuration a recognizable name
	NAME = "line_cross"

			# We use a GPU with 12GB memory, which can fit two images.
			# Adjust down if you use a smaller GPU.

			# Set batch size where, to process videos in parellel
			# Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 8

	# Number of classes (including background)
	NUM_CLASSES = 1 + 1  # Background + grape

	# Number of training steps per epoch
	STEPS_PER_EPOCH = 100

	# Skip detections with < 75% confidence
	DETECTION_MIN_CONFIDENCE = 0.9



def color_splash(image, mask_red):
	# Apply color in the zone of polygones according to the masks
	"""Apply color splash effect.
	image: RGB image [height, width, 3]
	mask: instance segmentation mask [height, width, instance count]

	Returns result image.
	"""
	# Copy color pixels from the original color image where mask is set
	red = image.copy()
	red[:, :, 1] = 0
	red[:, :, 2] = 0

	if mask_red.shape[-1] > 0:
		# We're treating all instances as one, so collapse the mask into one layer
		# Set the values of red or blue according to the boolean condition of the masks
		
		mask_red = (np.sum(mask_red, -1, keepdims=True) >= 1)
		# Set red pixels where the pixels in mask_red are 1 otherwise set image pixels
		splash = np.where(mask_red, red, image).astype(np.uint8)

	else:
		splash = image
	return splash


def process_red_result(result):

	red_result = {}  	

	red_result['red_masks'] = np.array([])
	red_result['red_rois'] = np.array([])
	red_result['red_scores'] = np.array([])
	# If exists at least 1 detection of a class we procede to extract their respective mask, roi, and score.
	if 1 in result['class_ids']:

		# changes the rows for channels to itereate over channels that represent a mask and verify if this mask is for class 1
		# then, it returns the np array back to the normal format 

		red_result['red_masks'] =\
			np.array([v for j, v in enumerate(np.swapaxes(result['masks'], 0, 2)) if result['class_ids'][j] == 1]) 
		red_result['red_masks']=np.swapaxes(red_result['red_masks'],0,2)

		red_result['red_rois'] =\
			np.array([v for j, v in enumerate(result['rois']) if result['class_ids'][j] == 1])
		
		red_result['red_scores'] =\
			np.array([v for j, v in enumerate(result['scores']) if result['class_ids'][j] == 1])

	return red_result
					

def detect_and_color_splash(model):
	# Variables to get the proccessing durations
	start_time = time.time()

	# Get the file names information
	input_name = os.path.basename(os.path.normpath(args.video))
	base_name = input_name.split(".")[0]
	_, campo_id, cuartel, hilera_id, ampm = base_name.split("_")
	
	print("[INFO] Input Video File: ",input_name)

	# Image or video?
	if args.video:
		# Video capture
		vcapture = cv2.VideoCapture(args.video)
		width = VIDEO_CAPTURE_WIDTH
		height = VIDEO_CAPTURE_HEIGHT
		fps = vcapture.get(cv2.CAP_PROP_FPS)

		print("[INFO] FPS from original video: {:.2f}".format(fps)) 

		# Define codec and create video writer 
		file_name = base_name + "_prediction.avi"
		vwriter = cv2.VideoWriter(os.path.join(args.output_dir,file_name),
								cv2.VideoWriter_fourcc(*'MJPG'),
								fps, (width, height))

		# Define variables to record the metadata
		racimo_locations = {}
		predictions_output = []
		current_distance = 0

		# Get the total frames in the current video
		totalFrames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT)) 
		
		# Define a images variable to perform detections with simultaneous images
		images = []
		simultaneous_images = model.config.IMAGES_PER_GPU

		for frameCount in range(totalFrames):		
			#Increase the distance
			current_distance += DISTANCE_PER_FRAME

			#Print the curretn frame
			print("[INFO] Frame: {}".format(frameCount))

			current_time = datetime.datetime.now() 
			success, image = vcapture.read() 
 
			if success == False: continue 

			# resize the current frame
			image = cv2.resize(image, (width, height)) 
						
			if success:
				# OpenCV returns images as BGR, convert to RGB 
				image = image[..., ::-1] 
				images.append(image)

				# Process the prediction with simultaneous images condition
				if len(images) != simultaneous_images: 
					continue

				# Detect objects # list de dictionaries
				results = model.detect(images, verbose=0) 
				
				for i, result in enumerate(results): 
				# Dataset division into two classes 
					if(len(result['masks'])==0): 
						continue 
					else:
						result = process_red_result(result)	
					
					# Color splash
					splash = color_splash(images[i], result['red_masks'])
					# splash = images[i]

					# Generate bbox_xywh from detection red
					detections_red = result["red_rois"] 	#[N, (y1, x1, y2, x2)]
					bbox_xywh_red = np.zeros((detections_red.shape[0],4))

					if(detections_red is not [] and len(detections_red)!=0):

						# Get the bbox points according YOLO convention
						bbox_xywh_red[:, 0] = (detections_red[:, 1] + detections_red[:, 3])/2	#x_center
						bbox_xywh_red[:, 1] = (detections_red[:, 0] + detections_red[:, 2])/2	#y_center
						bbox_xywh_red[:, 2] = (detections_red[:, 3] - detections_red[:, 1])		#width
						bbox_xywh_red[:, 3] = (detections_red[:, 2] - detections_red[:, 0])		#height

						# Put the prediction score in the bbox
						cls_conf_red = result['red_scores']
						for i in range(len(detections_red)):
							cv2.putText(splash, str(result['red_scores'][i]), 
													(int(bbox_xywh_red[i][0]), int(bbox_xywh_red[i][1])), 
													cv2.FONT_HERSHEY_PLAIN, 
													1, 
													[255, 255, 255], 
													1)
						
						# Update the tracker of red color
						outputs_red = deepsort_red.update(bbox_xywh_red, cls_conf_red, splash)

						if len(outputs_red) > 0:
							# Draw the tracker box # output red row [x1,y1,x2,y2,id_racimo]
							bbox_xyxy_red = outputs_red[:, :4]
							identities_red = outputs_red[:, -1] 
							splash = draw_boxes(splash, bbox_xyxy_red, identities_red, class_label="Uva")

							# Get the racimo ids if it is not include in the dict and crosses the line
							# with its respective frame Count and x_center
							for row in outputs_red:
								id_racimo = row[4]

								#Detect if the grape bunches are in the other side of the line
								# if VIDEO_DIRECTION == 'left':
								# 	xmin = row[0]
								# 	cross_line = xmin > X_LINE
								# elif VIDEO_DIRECTION == 'right':
								# 	xmax = row[2]
								# 	cross_line = xmax < X_LINE

								#Detect if the grape bunches cross the line
								xmin = row[0]
								xmax = row[2]
								cross_line = xmin < X_LINE and xmax > X_LINE

								#Only accumulate the no-repeated id_racimo when it crosses the line
								if(id_racimo not in racimo_locations) and cross_line:
									racimo_locations.update({id_racimo : (frameCount,(row[2]-row[0])/2.0, current_distance)})

							# Get prediction info
							for identity in identities_red:								
								area = deepsort_red.tracker.get_area_by_id(identity)
								if area >= 0:
									# print(str(deepsort.tracker.get_area_by_id(identity)))
									predictions_output.append((campo_id, cuartel,hilera_id,ampm, int(identity), int(area)))

					# Draw the line in the splash
					cv2.line(splash, (X_LINE, 0), (X_LINE, height), (0, 255, 255), 2)

					# Display the counted text box
					label_red = "Conteo de racimos: {}".format(len(racimo_locations.items()))
					splash = draw_text_area(splash,label_red, (width,height) )

					# Display the line in the current frame and distance in meters
					str_distance = "Distance: {:.2f} (m)".format(float(current_distance) / 100.0)
					splash = draw_text_area(splash,str_distance,(width,height), location='left-bottom', text_color=[0, 0, 255], thickness = 1)

					# RGB -> BGR to save image to video
					splash = splash[..., ::-1]

					# Add image to video writer
					vwriter.write(splash)
          
				images = []
				time_per_frame = (datetime.datetime.now() - current_time)/simultaneous_images
				fps_w = simultaneous_images/(datetime.datetime.now() - current_time)
				print("[INFO] Time per Frame: {} FPS: {:.2f}.".format( time_per_frame, fps_w))

		vwriter.release()
		print("[INFO] Saving Video File")

		# Create Pickles
		pickle_name = "prediction_" + str(campo_id) + "_" + str(cuartel) + "_" +str(hilera_id)+ "_"+str(ampm)		
		predictions_dir = os.path.join(args.pickles_dir,"prediction_pickles")

		with open(os.path.join(predictions_dir, pickle_name + '.pkl'), 'wb') as f:
			pickle.dump(predictions_output, f)

		print("[INFO] Saving Pickles Predictions")

		pickle_loc_name = "locations_" + str(campo_id) + "_" + str(cuartel) + "_" +str(hilera_id)+ "_"+str(ampm)
		locations_dir = os.path.join(args.pickles_dir,"location_pickles")

		with open(os.path.join(locations_dir, pickle_loc_name + '.pkl'), 'wb') as f:
			pickle.dump(racimo_locations, f)

		print("[INFO] Saving Pickles Locations")

		final_time = time.time()
		duration = final_time - start_time
		print("[INFO] Total Proccessing Duration Time: {}".format(datetime.timedelta(seconds=int(duration))))


def create_dirs():
	# Create the needed dirs if these does not exists
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	output_prediction = os.path.join(args.pickles_dir,"prediction_pickles")
	if not os.path.exists(output_prediction):
		os.makedirs(output_prediction)

	output_location = os.path.join(args.pickles_dir,"location_pickles")
	if not os.path.exists(output_location):
		os.makedirs(output_location)

############################################################
#  Inference
############################################################

if __name__ == '__main__':

	# Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Train Mask R-CNN to detect grapes.')
	parser.add_argument('--weights', required=True,
						metavar="/path/to/weights.h5",
						help="Path to weights .h5 file or 'coco'")
	parser.add_argument('--video', required=False,
						metavar="path or URL to video",
						help='Video to apply the color splash effect on')
	parser.add_argument('--logs', required=False,
						default=DEFAULT_LOGS_DIR,
						metavar="/path/to/logs/",
						help='Logs and checkpoints directory (default=logs/)')						
	parser.add_argument('--pickles_dir', required=False,
						default="stuff/pickles/TestCampo",
						metavar="path to the metada",
						help='Path to store the pickles files')
	parser.add_argument('--output_dir', required=False,
						default="stuff/output_videos/TestCampo",
						metavar="path to the ouput video",
						help='Path to the output video with the predictions')
	args = parser.parse_args()

	create_dirs()

	cfg = get_config()
	cfg.merge_from_file("./deep_sort_pytorch/configs/yolov3.yaml")
	cfg.merge_from_file("./deep_sort_pytorch/configs/deep_sort.yaml")
	deepsort_blue = build_tracker(cfg, use_cuda=1)
	deepsort_red = build_tracker(cfg, use_cuda=1)


	# Validate arguments
	assert args.video, "Provide --video to apply color splash"

	print("Weights: ", args.weights)
	print("Logs: ", args.logs)
	
	config = InferenceConfig()
	config.display() #Display the configuration values

	# Create model
	model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
	
	# Select weights file to load
	if args.weights.lower() == "coco":
		weights_path = COCO_WEIGHTS_PATH
		# Download weights file
		if not os.path.exists(weights_path):
			utils.download_trained_weights(weights_path)
	elif args.weights.lower() == "last":
		# Find last trained weights
		weights_path = model.find_last()
	elif args.weights.lower() == "imagenet":
		# Start from ImageNet trained weights
		weights_path = model.get_imagenet_weights()
	else:
		weights_path = args.weights

	# Load weights
	print("Loading weights ", weights_path)
	if args.weights.lower() == "coco":
		# Exclude the last layers because they require a matching
		# number of classes
		model.load_weights(weights_path, by_name=True, exclude=[
			"mrcnn_class_logits", "mrcnn_bbox_fc",
			"mrcnn_bbox", "mrcnn_mask"])
	else:
		model.load_weights(weights_path, by_name=True)

	# Execute the inference process
	print("Starting Inference Process")
	detect_and_color_splash(model)

