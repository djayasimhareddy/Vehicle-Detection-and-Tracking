from vehicle_detector import VehicleDetectionSystem
from ui import create_gui_menu

def main():
    det = VehicleDetectionSystem()
    choice, path = create_gui_menu()
    if choice == "webcam":
        det.webcam_detection()
    elif choice == "video":
        det.video_detection(path)
    elif choice == "image":
        det.image_detection(path)
    print("✅ Session Complete.")

if __name__ == "__main__":
    main()
