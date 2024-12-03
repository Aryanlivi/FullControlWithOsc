from enum import Enum
class OutputMethod(Enum):
    OSC=1
    PyAutoGUI=2

#DATASET_DIR = 'Single_person_violent'
DATASET_DIR = 'Final_Dataset_trim'

# Action List
CLASSES_LIST = ['Jump',
 'Left High Kick',
 #'Idle',
 'Punch Left',
 #'Uppercut',
 'Block',
 'Duck',
 #'Stance',
 'Punch Right',
 'Right Low Kick',
 'Right High Kick',
 'Left Low Kick']

# Directory to save csv files
OUTPUT_DIR = 'Output'

MODEL='model2v2.h5'

# Dataframe column header for a single video
custom_headers = ['Frame Number', 'x','y','z','Visibility']

# Define predefined parameters of the model
max_num_frames = 15  # maximum number of frames per action sequence
num_landmarks = 23    # number of landmarks per frame
num_features = 4      # number of features per landmark (x, y, z, visibility)
num_classes = len(CLASSES_LIST)       # number of action classes

PREDICTION_CONFIDENCE=0.8
DEBOUNCE_TIME=0.6
exclude_points = [1,2,3,4,5,6,7,8,9,10]