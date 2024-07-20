import cv2
import pyautogui
import time
import mediapipe as mp
import csv
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def trigger_key_press(key_name):
    pyautogui.press(key_name)


# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#DATASET_DIR = 'Single_person_violent'
DATASET_DIR = 'Final_Dataset_trim'

# Action List
CLASSES_LIST = ['Jump',
 'Left High Kick',
 'Idle',
 'Punch Left',
 'Uppercut',
 'Block',
 'Duck',
 'Stance',
 'Punch Right',
 'Right Low Kick',
 'Right High Kick',
 'Left Low Kick']

# Directory to save csv files
OUTPUT_DIR = 'Output'

# Dataframe column header for a single video
custom_headers = ['Frame Number', 'x','y','z','Visibility']

# Load the model
model = tf.keras.models.load_model('model.h5')

# Define predefined parameters of the model
max_num_frames = 20  # maximum number of frames per action sequence
num_landmarks = 33    # number of landmarks per frame
num_features = 4      # number of features per landmark (x, y, z, visibility)
num_classes = len(CLASSES_LIST)       # number of action classes

# Extract landmarks
def preprocess_frame(results):
    # Extract landmarks and preprocess
    landmarks = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
    return landmarks

def predict_action(sequence):
    # Pad the sequence
    padded_sequence = pad_sequences([sequence], maxlen=max_num_frames, dtype='float32', padding='post', truncating='post', value=-1)
    padded_sequence = padded_sequence.reshape((padded_sequence.shape[0], padded_sequence.shape[1], num_landmarks * num_features))
    
    # Predict the action
    predictions = model.predict(padded_sequence)
    return np.argmax(predictions), np.max(predictions)

# Normalize landmarks
def normalize_z_values(landmarks):
    """
    Normalize the z-values of pose landmarks to a range of [0, 1].

    Parameters:
    landmarks (list): List of pose landmarks, each represented as [x, y, z, visibility].

    Returns:
    list: Normalized pose landmarks with z-values scaled to [0, 1].
    """
    if not landmarks: return landmarks

    min_z = -2
    max_z = 2

    
    # Normalize z-values to range [0, 1]
    normalized_landmarks = []
    for lm in landmarks:
        if max_z - min_z != 0:
            normalized_z = (lm[2] - min_z) / (max_z - min_z)
        else:
            normalized_z = lm[2]  # Handle edge case
        # Replace with the desired visibility value
        
        normalized_landmarks.append([lm[0], lm[1], normalized_z, lm[3]])  # Keep x, y, visibility unchanged
        # else:
        #     normalized_landmarks.append([-1,-1,-1,-1])

    return normalized_landmarks


# Start capturing video from webcam
cap = cv2.VideoCapture(0)
# Initialize a sequence to store landmarks
sequence = []
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Perform pose detection
        results = pose.process(image)

        # Convert the image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        
        # Draw the pose annotation on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract landmarks and preprocess
            landmarks = preprocess_frame(results)
            normalized_landmarks = normalize_z_values(landmarks)
            
            # write_landmarks_to_csv_temp(normalized_landmarks, framess, csv_data2)
            if len(normalized_landmarks) > 0:
                sequence.append(normalized_landmarks)
            
            # If sequence is too long, remove the oldest frame
            if len(sequence) > max_num_frames:
                sequence.pop(0)
            #cv2.putText(image, f'{len(sequence)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Make a prediction if sequence has enough frames
            if len(sequence) == max_num_frames:
                action_idx, confidence = predict_action(sequence)
                action_label = CLASSES_LIST[action_idx]
                
                # Display the predicted action on the frame
                if confidence> 0.7:
                    cv2.putText(image, f'Action: {action_label} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    if action_label == 'Punch Right':
                        print('Punch detected! Pressing p key')
                        trigger_key_press('p')
                    if action_label == 'Punch Left':
                        trigger_key_press('o')
                    if action_label == 'Left High Kick' or action_label == 'Right High Kick':
                        trigger_key_press('k')

        
        # Display the resulting frame
        cv2.imshow('Webcam Action Recognition', image)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # with open(f'outfile', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     # Write each row of the 2D array
    #     for row in csv_data2:
    #         writer.writerow(row)
    # Release the capture
    cap.release()
    cv2.destroyAllWindows()