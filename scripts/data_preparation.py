import os
import pickle
import time
import sys
import feature_extraction

# Minimum number of images per class
min_nrof_image_per_class = 10

model_dir = '../model'

# Name of model to train and evaluate
dataset_filename = os.path.join(model_dir,'dataset.pkl')

def main():
    start_time = time.time()
    labels, embeddings = _prepare_dataset('../preprocessed_data')
    with open(dataset_filename, 'wb') as outfile:
        pickle.dump((labels, embeddings), outfile)
    print('\n\nCompleted in {} seconds'.format(time.time() - start_time))
    print('Dataset size: {}'.format(len(labels)))


# brief : Load previously saved dataset
# return:   labels: class names (y) eg. John Doe
#           embeddings: facial encodings of subject (x)
def load_dataset():
    with open(dataset_filename, 'rb') as f:
        labels, embeddings = pickle.load(f)
        return labels, embeddings


# brief :   Shows the current progress of the program
# param :   count: an int that represents current progress
#           total: an int that represents total work to be done
#           status: a string that describe the current progress
def _progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%50s' % (bar, percents, '%', status))
    sys.stdout.flush()


# brief :   Read the images in input dir and format it nicely into a list
# param :   input_dir: absolute path to the input directory
# return:   labels: class names (y) eg. John Doe
#           embeddings: facial encodings of subject (x)
def _prepare_dataset(input_dir):
    current = 0
    total = _get_total_image(input_dir)
    labels = []
    embeddings = []

    for dir in os.listdir(input_dir):
        # if invalid directory, ignore
        if dir.startswith('.'):
            continue
        
        # loop through valid directory
        for image_path in os.listdir(os.path.join(input_dir, dir)):
            # if less than minimum images per class, ignore
            if len(os.listdir(os.path.join(input_dir,dir))) <min_nrof_image_per_class:
                continue

            # if invalid image_path, ignore
            if image_path.startswith('.') or not image_path.endswith('.jpg'):
                continue

            # form absolute path to image
            full_path_to_image = os.path.join(input_dir, dir)
            full_path_to_image = full_path_to_image + '/' + image_path
            # update progress
            current += 1
            status = str(current) + '/' + str(total) + ' ' + image_path
            _progress(current, total, status=status)
            # extract facial encoding
            face_encodings = feature_extraction.get_face_encodings(full_path_to_image)
            if len(face_encodings) < 1:
                continue
            # append into array
            embeddings.append(face_encodings[0])
            labels.append(dir)

    return labels, embeddings


# brief :   Calculate the total number of images to process, purpose to track progress
# param :   input_dir: the absolute path to the training data directory
# return:   an int that represents the total images to process
def _get_total_image(input_dir):
    total = 0
    for dir in os.listdir(input_dir):
        if not dir.startswith('.'):
            dir_path = os.path.join(input_dir,dir)
            if len(os.listdir(dir_path)) < min_nrof_image_per_class:
                continue
            total += len(os.listdir(dir_path))
    return total

if __name__ == '__main__':
    main()