import numpy as np
import cv2
import dlib
import os

model_dir = '../model'

# Declaring dlib models to detect face and generate face encodings
shape_predictor = dlib.shape_predictor(os.path.join(model_dir,'shape_predictor_68_face_landmarks.dat'))
face_recognition_model = dlib.face_recognition_model_v1(os.path.join(model_dir,'dlib_face_recognition_resnet_model_v1.dat'))
face_detector = dlib.get_frontal_face_detector()

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def extract_feature(input_path, output_path):
	image = cv2.imread(input_path)
	detected_faces = face_detector(image,1)
	shaped_faces = [shape_predictor(image,face) for face in detected_faces]
	face_encodings = [shape_to_np(face_pose) for face_pose in shaped_faces]
	for face_encoding in face_encodings:
		for (x,y) in face_encoding:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	cv2.imwrite(output_path, image)
	