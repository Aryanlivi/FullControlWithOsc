# from pythonosc import udp_client
from webcamHandler import WebcamHandler

# # Initialize OSC client.
# client = udp_client.SimpleUDPClient("127.0.0.1", 39539)

class PoseDetectOSC(WebcamHandler):
    def __init__(self,camera_index=0):
        super().__init__(camera_index)
        self.start(use_mp=True)
    def detect_pose(self):
        pass
        
osc_detect=PoseDetectOSC()
osc_detect.start(use_mp=True)