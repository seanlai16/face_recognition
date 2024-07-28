import dlib
import numpy as np
import os
import pickle
import cv2


# Hardcoded the directory path to models
# Change this path to your model directory
model_dir = '../model'

# Hardcoded the directory path for testing
# Change this path to test your own images
input_dir = '../test'

# Declaring dlib models to detect face and generate face encodings
shape_predictor = dlib.shape_predictor(os.path.join(model_dir,'shape_predictor_68_face_landmarks.dat'))
face_recognition_model = dlib.face_recognition_model_v1(os.path.join(model_dir,'dlib_face_recognition_resnet_model_v1.dat'))
face_detector = dlib.get_frontal_face_detector()

# Name of the predictor model previously trained
classifier_file = os.path.join(model_dir,'classifier.pkl')

def classify(input_path):
    emb_array  = get_face_encodings(input_path)

    # Read model
    with open(classifier_file, 'rb') as f:
        # Load the data inside the model
        model, class_names = pickle.load(f)
        # Make predictions on the face encodings
        predictions = model.predict_proba(emb_array)

        best_class_indices = np.argmax(predictions, axis=1)

        return class_names[best_class_indices[0]]


# brief :   Generates face encodings of an image
# param :   path_to_image: absolute path to an image
# return:   face_encodings: a list of np array storing face encodings in 128 dimensions
#           test_img: a copy of the target image in cv2 format, purpose: to not overwrite original image
#           detected_faces: a list of Dlib Detection Object that provides the position of the faces
def get_face_encodings(path_to_image):
    image = cv2.imread(path_to_image)
    test_img = image.copy()

    detected_faces = face_detector(image,1)

    shaped_faces = [shape_predictor(image,face) for face in detected_faces]

    face_encodings = [np.array(face_recognition_model.compute_face_descriptor(image,face_pose,1)) for face_pose in shaped_faces]

    return face_encodings
