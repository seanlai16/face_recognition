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


# brief :   Make predictions using the previously trained model & indicate names in picture
# param :   emb_array: a list of face_encodings
#           classifier_filename: the dir to the pre-trained model
#           image: a cv2 image
#           rects: Dlib Detection Object
def predict_classifier(emb_array, classifier_filename, image, rects):
    print('predicting...')
    # IF check whether model exists
    if not os.path.exists(classifier_filename):
        raise ValueError('Classifier Not Found!')
    # ENDIF

    # Read model
    with open(classifier_filename, 'rb') as f:
        # Load the data inside the model
        model, class_names = pickle.load(f)
        # Make predictions on the face encodings
        predictions = model.predict_proba(emb_array)

        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

        for index,rect in enumerate(rects):
            # Draw rectangle over faces detected
            draw_rectangle(image, rect)

            # Draw name of faces
            if best_class_probabilities[index] <0.5:
                draw_text(image, '?????', rect)
            else:
                text = class_names[best_class_indices[index]]
                draw_text(image,text,rect)

        # Show the image & wait for keystroke
        cv2.imshow("",image)
        cv2.waitKey(0)


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

    return face_encodings, test_img, detected_faces


# brief :   Draw a rectangle over a given face position
# param :   img: a cv2 image
#           rect: a Dlib Detection Object
def draw_rectangle(img, rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)


# brief :   Draw a text above a given face position, purpose: to indicate name
# param :   img: a cv2 image
#           text: string to write above the given face
#           rect: a Dlib Detection Object
def draw_text(img, text, rect):
    x = rect.left()
    y = rect.top()
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),2)


# MAIN
# brief: loops through the input directory and predict all images in it
for image in os.listdir(input_dir):
    # IF to skip system files such as .DStore
    if not image.startswith('.'):
        # IF cv2 only works on jpg files
        if image.endswith('.jpg') or image.endswith('.jpeg'):
            # Get the absolute path of the image
            full_path_to_image = os.path.join(input_dir,image)
            print(full_path_to_image)
            # Generate facial encodings
            face_encodings, image, rects = get_face_encodings(full_path_to_image)
            # Make predictions & draw rectangle
            predict_classifier(face_encodings,classifier_file, image, rects)
        # ENDIF
    # ENDIF

