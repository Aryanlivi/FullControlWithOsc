import cv2
from webcamHandler import WebcamHandler
from PredictPose import predict_action,preprocess_frame,normalize_z_values
from Constants import *
from OscMessageHandler import osc_message_handler
from AutoGuiHandler import autogui_message_handler
import time


# Initialize variables for tracking the last action
last_action = None
last_action_time = time.time()
readyToPredict = False
# action_label=' '
confidence=0
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
        global action_label,confidence, readyToPredict
        
        # Extract landmarks and preprocess
        #landmarks = preprocess_frame(results)
        temp_csv = []       
        for idx,landmark in enumerate(results.pose_landmarks.landmark):                
            if idx not in exclude_points:                            
                #print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
                temp_csv.append([landmark.x, landmark.y, landmark.z,landmark.visibility]) 
        normalized_landmarks = normalize_z_values(temp_csv)
        #normalized_landmarks = normalize_z_values(landmarks)
        if len(normalized_landmarks) > 0:
            self.sequence.append(normalized_landmarks)
            
        # If sequence is too long, remove the oldest frame
        if len(self.sequence) > max_num_frames:
            self.sequence.pop(0)

        visible_landmarks_count = 0
        # Calculate the number of visible landmarks. We skip 0 to 10 because they are facial points and are not required
        for idx,l in enumerate(results.pose_landmarks.landmark):
            if idx>10 and l.visibility > VISIBILITY_THRESHOLD:
                visible_landmarks_count += 1

        # Only draw the landmarks if more that 20 landmarks are detected
        if visible_landmarks_count < VISIBLE_POINTS_REQUIREMENT: 
            if readyToPredict:
                print(f'Only {visible_landmarks_count} points visisble.')
            readyToPredict= False
        else:
            if not readyToPredict:
                print(f'Ready to Predict Visible points: {visible_landmarks_count}')
            readyToPredict = True
            #cv2.putText(image, f'Only {visible_landmarks_count} points visible', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)                         
        
        
        # Make a prediction if sequence has enough frames
        if len(self.sequence) == max_num_frames and readyToPredict:
            action_idx, confidence = predict_action(self.sequence)
            action_label = CLASSES_LIST[action_idx]
            if confidence>0.8:
                self.sequence = [] 
                osc_message_handler(action_label)
                # autogui_message_handler(action_label) 
            # Display the predicted action on the frame
        if confidence> 0.8:
            cv2.putText(image_bgr, f'Action: {action_label} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #

            
    def make_prediction(self,image_bgr):
        global last_action, last_action_time
        action_idx, confidence = predict_action(self.sequence)
        action_label = CLASSES_LIST[action_idx]
        
        # Display the predicted action on the frame
        if confidence> PREDICTION_CONFIDENCE:
            cv2.putText(image_bgr, f'Action: {action_label} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            current_time=time.time()
            if action_label != last_action or (current_time - last_action_time) > DEBOUNCE_TIME:
                if self.input_method==OutputMethod.OSC:
                    osc_message_handler(action_label)
                elif self.input_method==OutputMethod.PyAutoGUI:
                    autogui_message_handler(action_label)
                else:
                    print("invalid input method")
                last_action=action_label
                last_action_time=current_time

        
