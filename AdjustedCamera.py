import numpy as np
import cv2

# Load the calibration data
calibration_data = np.load('calibration_data.npz')
mtx = calibration_data['mtx']
dist = calibration_data['dist']

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

# Set the resolution to 1280x720
cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Undistort the frame using the calibration data
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)

    # Display the undistorted frame
    cv2.imshow('Undistorted Video', undistorted_frame)

    # Press 'ESC' to quit the loop
    if cv2.waitKey(1) == 27:  # 27 is the ASCII code for 'ESC'
        break

# When everything done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
