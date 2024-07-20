import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Constants import *
import numpy as np

# Load the model
model = tf.keras.models.load_model(MODEL)

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

