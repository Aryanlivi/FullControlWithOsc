from pythonosc import udp_client
# Initialize OSC client.
client = udp_client.SimpleUDPClient("127.0.0.1", 39539)

# Define OSC ADDRESSES
BASE_ADDRESSES = {
    'Punch Right': '/action/punch_right',
    'Punch Left': '/action/punch_left',
    'Right High Kick': '/action/kick',
    'Left High Kick': '/action/kick',
    'Stance':'action/stance',
    'Right Low Kick':'/action/kick',
    'Left Low Kick':'/action/kick',
    'Duck':'/action/duck',
    'Block':'/action/block'
}

ACTION_IDENTIFIERS = {
    'Punch Right': 1,
    'Punch Left': 2,
    'Right High Kick': 1,
    'Left High Kick': 2,  
    'Stance':1,
    'Right Low Kick':1,
    'Left Low Kick':2,
    'Duck':1,
    'Block':1
}

def osc_message_handler(action_label):
    base_address=BASE_ADDRESSES.get(action_label)
    value=ACTION_IDENTIFIERS.get(action_label)
    
    if base_address and value:
        client.send_message(base_address, value)  # Send message with value 1
        print(f"Sent OSC message: {base_address}/{value}")
    else:
        print(f"No OSC address defined for action: {action_label}")
    