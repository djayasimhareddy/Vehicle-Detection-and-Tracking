# main.py
from detection_system import VehicleDetectionSystem
from gui import create_gui_menu, console_menu

def main():
    system = VehicleDetectionSystem()
    try:
        mode, path = create_gui_menu()
    except:
        mode, path = console_menu()

    if mode == 'webcam':
        system.detect_webcam()
    elif mode == 'video':
        system.detect_video(path)
    elif mode == 'image':
        system.detect_image(path)
    else:
        print("Goodbye.")

if __name__ == "__main__":
    main()

