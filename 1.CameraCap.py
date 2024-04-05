import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)  # '0' is the default value for the primary camera

# Set the resolution
desired_width = 640
desired_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Counter for the image name
counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Enter to Capture, Esc or Q to Quit', frame)

    # Wait for the Enter or Esc or Q key to be pressed
    key = cv2.waitKey(1)
    if key == 13:  # 13 is the Enter Key
        img_name = f"opencv_frame_{counter}_{desired_height}p.png"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        counter += 1

    # Exit loop if 'q' or 'Q' or 'ESC' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q') or key == ord('Q'):
        break
# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
