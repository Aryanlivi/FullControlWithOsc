from OscMessageHandler import osc_message_handler
import time

prev_action_label = ''
prev_action_time = time.time()

def actionHandler(current_action_label):
    global prev_action_label, prev_action_time
    current_action_time = time.time()    
    time_delta = current_action_time - prev_action_time
    if current_action_label == prev_action_label and time_delta < 2:
        return
    osc_message_handler(current_action_label)
    prev_action_time = current_action_time
    prev_action_label = current_action_label
    
