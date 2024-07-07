import cv2
import Yolo as yolo

def main():
    # Initialize camera capture
    cap = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set camera resolution
    cap.set(3, 1280)
    cap.set(4, 720)

    # Initialize ShopIA instance
    class_yolo = yolo.Yolo()

    # Start the main loop
    while True:
        # Run object detection and display
        class_yolo.yolo_run(cap)

        # Ask user if they want to continue
        key = input("Press 'q' to quit, or any other key to continue: ")
        if key.lower() == 'q':
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

# Entry point of the script
if __name__ == "__main__":
    main()
