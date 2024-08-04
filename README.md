# face_recognition

## Setup

1. create python virtual env `python3 -m venv venv`
2. activate env `source venv/bin/activate` in macos OR `venv\Scripts\activate` in windows
3. run `pip install -r requirements.txt`
4. if pip complain cmake, run `pip install cmake` in system/global (outside virual env)

## How to use

1. run `python preprocess.py` to preprocess the training set in `/training_data`
2. run `python data_preparation.py` to extract features from image and save as labels & facial embeddings
3. run `python training_validation.py` to train and validate model
4. run `python gui.py` to open GUI to test the trained model on unseen data
5. run `python visualisation.py` to visualise feature space in 3D