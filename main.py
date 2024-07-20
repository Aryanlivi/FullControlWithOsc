from PoseHandler import PoseHandler
from Constants import OutputMethod

def main(input_method):
    if input_method == OutputMethod.OSC:
        print("osc")
        pose_handler=PoseHandler(input_method=OutputMethod.OSC)
    elif input_method == OutputMethod.PyAutoGUI:
        print("gui")
        pose_handler=PoseHandler(input_method=OutputMethod.PyAutoGUI)
    else:
        print("Invalid input method")


if __name__ == "__main__":
    selected_method = OutputMethod.OSC  # or OutputMethod.PyAutoGUI
    main(selected_method)