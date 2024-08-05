import argparse
import glob
import logging
import multiprocessing as mp
import os
import time
import dlib
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Hardcoded the model dir
model_dir = '../model/'

# Declaring dlib models to detect face and generate face encodings
shape_predictor = dlib.shape_predictor(os.path.join(model_dir,'shape_predictor_68_face_landmarks.dat'))
face_recognition_model = dlib.face_recognition_model_v1(os.path.join(model_dir,'dlib_face_recognition_resnet_model_v1.dat'))
face_detector = dlib.get_frontal_face_detector()

# brief :   Generates face encodings of an image
# param :   path_to_image: absolute path to an image
# return:   face_encodings: a list of np array storing face encodings in 128 dimensions
def get_face_encodings(path_to_image):
	# Read image from path
    image = cv2.imread(path_to_image)
	# Detect face
    detected_faces = face_detector(image,1)
	# Detect facial landmarks for each face
    shaped_faces = [shape_predictor(image,face) for face in detected_faces]
	# Compute 128-D facial vectors for each face
    face_encodings = [np.array(face_recognition_model.compute_face_descriptor(image,face_pose,1)) for face_pose in shaped_faces]
    return face_encodings

# brief :   Converts 68 facial landmarks into (x, y) coordinates in the form of np array
# param :   shape: shape object from dlib
# return:   coords: np array of (x, y) coordinates of facial landmarks
def _shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

# brief :   Extract facial landmarks from input image. Plot and save as new image
# param :   input_path: input image path
#           output_path: output image path
# return:   -
def plot_landmarks(input_path, output_path):
	image = cv2.imread(input_path)
	detected_faces = face_detector(image,1)
	shaped_faces = [shape_predictor(image,face) for face in detected_faces]
	face_encodings = [_shape_to_np(face_pose) for face_pose in shaped_faces]
	for face_encoding in face_encodings:
		for (x,y) in face_encoding:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	cv2.imwrite(output_path, image)