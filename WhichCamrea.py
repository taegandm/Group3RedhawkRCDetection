import cv2

# Function to check if the camera at the given index is available
def check_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use DirectShow (via videoInput)
    if cap is None or not cap.isOpened():
        return False
    cap.release()
    return True

# List to hold camera indices and names
cameras = []

# Loop through camera indices to find available cameras
for i in range(10):  # Increase the range if you have more cameras
    if check_camera(i):
        cameras.append((i, "Camera #"+str(i)))  # Camera name/make is not directly accessible via OpenCV

# Print the available cameras
if cameras:
    print("Available cameras:")
    for index, name in cameras:
        print(f"Index: {index}, Name/Make: {name}")
else:
    print("No cameras detected.")
