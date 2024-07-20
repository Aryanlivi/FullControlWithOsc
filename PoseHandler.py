import cv2
from webcamHandler import WebcamHandler
from PredictPose import predict_action,preprocess_frame,normalize_z_values
from Constants import *
from OscMessageHandler import osc_message_handler
from AutoGuiHandler import autogui_message_handler

class PoseHandler(WebcamHandler):
    def __init__(self,input_method=OutputMethod.OSC): 
        super().__init__()
        self.sequence=[]
        self.input_method=input_method
        self.start()
        
    def start(self):
        self.init_mp()
        while self.webCam.isOpened():
            ret,frame=self.webCam.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            self.draw_landmarks(frame)
            
            # Break the loop on 'q' key press.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.release()
        
    def draw_landmarks(self, frame):
        results,image_bgr=self.process_frame(frame)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image_bgr,results.pose_landmarks,self.mp_pose.POSE_CONNECTIONS)
            self.preprocess_landmarks(results,image_bgr) 
        if self.input_method==OutputMethod.OSC:
            self.display(image_bgr,"OSC Webcam")
        elif self.input_method==OutputMethod.PyAutoGUI:
            self.display(image_bgr,"PyAutoGUI Webcam")
        else: 
            print("Invalid Input method")
        
    def preprocess_landmarks(self,results,image_bgr):
        # Extract landmarks and preprocess
        landmarks = preprocess_frame(results)
        normalized_landmarks = normalize_z_values(landmarks)
        if len(normalized_landmarks) > 0:
            self.sequence.append(normalized_landmarks)
            
        # If sequence is too long, remove the oldest frame
        if len(self.sequence) > max_num_frames:
            self.sequence.pop(0)
            
        # Make a prediction if sequence has enough frames
        if len(self.sequence) == max_num_frames:
            self.make_prediction(image_bgr)
            
    def make_prediction(self,image_bgr):
        action_idx, confidence = predict_action(self.sequence)
        action_label = CLASSES_LIST[action_idx]
        
        # Display the predicted action on the frame
        if confidence> PREDICTION_CONFIDENCE:
            cv2.putText(image_bgr, f'Action: {action_label} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if self.input_method==OutputMethod.OSC:
                osc_message_handler(action_label)
            elif self.input_method==OutputMethod.PyAutoGUI:
                autogui_message_handler(action_label)
            else:
                print("invalid input method")

