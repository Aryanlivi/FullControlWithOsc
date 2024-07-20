import pyautogui

def trigger_key_press(key_name):
    pyautogui.press(key_name)


def autogui_message_handler(action_label):
    if action_label == 'Punch Right':
        print('Punch detected! Pressing p key')
        trigger_key_press('p')
    if action_label == 'Punch Left':
        trigger_key_press('o')
    if action_label == 'Left High Kick' or action_label == 'Right High Kick':
        trigger_key_press('k')