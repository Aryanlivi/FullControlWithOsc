from pythonosc import udp_client
# # Initialize OSC client.
# client = udp_client.SimpleUDPClient("127.0.0.1", 39539)
def osc_message_handler(action_label):
    if action_label == 'Punch Right':
        print('Punch detected! Pressing p key')
    if action_label == 'Punch Left':
        print('Punch left')
    if action_label == 'Left High Kick' or action_label == 'Right High Kick':
        print("kick")
    