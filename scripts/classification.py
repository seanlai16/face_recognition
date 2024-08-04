import dlib
import numpy as np
import os
import pickle
import feature_extraction

# Hardcoded the directory path to models
# Change this path to your model directory
model_dir = '../model'

# Declaring dlib models to detect face and generate face encodings
shape_predictor = dlib.shape_predictor(os.path.join(model_dir,'shape_predictor_68_face_landmarks.dat'))
face_recognition_model = dlib.face_recognition_model_v1(os.path.join(model_dir,'dlib_face_recognition_resnet_model_v1.dat'))
face_detector = dlib.get_frontal_face_detector()

# Name of the predictor model previously trained
classifier_file = os.path.join(model_dir,'classifier.pkl')

# brief :   Runs prediction using previously trained SVM model
# param :   input_path: path to input image
# return:   best_class: class name with the highest confidence
#           best_class_probability: confidence of best_class
def classify(input_path):
    face_encodings = feature_extraction.get_face_encodings(input_path)

    # Read model
    with open(classifier_file, 'rb') as f:
        # Load the data inside the model
        model, label_encoder = pickle.load(f)
        # Make predictions on the face encodings
        predictions = model.predict_proba(face_encodings)

        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        best_class = label_encoder.inverse_transform([best_class_indices[0]])[0]
        best_class_probability = best_class_probabilities[0]

        return best_class, best_class_probability
