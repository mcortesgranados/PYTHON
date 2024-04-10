import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(0)  # 0 represents the camera index (0 for the first camera)

# Check if the camera opened successfullyq
if not cap.isOpened():
    print("Error: Failed to open the camera.")
else:
    # Capture frames from the camera
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame was successfully captured
        if ret:
            # Display the frame in a window
            cv2.imshow('Camera', frame)

            # Check for key press (press 'q' to exit)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error: Failed to capture frame.")
            break

    # Release the VideoCapture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
