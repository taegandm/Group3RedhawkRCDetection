import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)  # '0' is the default value for the primary camera

# Set the resolution
desired_width = 1280
desired_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Counter for the image name
counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Enter to Capture, Esc to Quit', frame)

    # Wait for the Enter or Esc key to be pressed
    key = cv2.waitKey(1)
    if key == 13:  # 13 is the Enter Key
        img_name = f"opencv_frame_{counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        counter += 1

    # Break the loop with the 'Esc' key
    if key == 27:  # 27 is the Esc Key
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
