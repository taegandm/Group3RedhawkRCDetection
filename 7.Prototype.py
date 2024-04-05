import numpy as np
import cv2
import serial
import time
import serial.tools.list_ports

# Find the serial port
def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        print(p)
        return p[0]
    return None

arduino_port = find_arduino_port()

# Connect the Pi with Arduino via serial port
if arduino_port:
    try:
        ser = serial.Serial(arduino_port, 9600, timeout=1)
        print(f"Connected to Arduino on {arduino_port}")
        ser.flush()
        # You can now use 'ser' to communicate with your Arduino
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
else:
    print("Arduino not found")


# Function for Speed and Steering control
# 90 for netural or stop
# Speed: >90 forward, 100 recommand
# Steering: >90 right, <90 left
def send_command(steering, velocity):
    command = f"{steering},{velocity}\n"
    ser.write(command.encode('utf-8'))

# Function to determine if the object is on the left or right side of the screen
def left_or_right(frame_width, dst):
    # Calculate the centroid of the quadrilateral
    centroid_x = np.mean(dst[:, 0, 0])
    # Determine the side based on the centroid's x-coordinate
    if centroid_x > frame_width / 2:
        return "Right"
    elif centroid_x < frame_width / 2:
        return "Left"
    
# Function to calculate the horizontal distance to the midline
def distance_to_midline(frame_width, dst):
    # Calculate the centroid of the quadrilateral
    centroid_x = np.mean(dst[:, 0, 0])
    # Calculate the distance to the midline
    distance = np.abs(centroid_x - frame_width / 2)
    return distance

# Load previously saved data from the .npz file
with np.load('calibration_data.npz') as X:
    camera_matrix, dist_coeffs = [X[i] for i in ('mtx', 'dist')]

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Load the template image
template_image = cv2.imread('Redhawk.png', 0)  # Ensure the image is grayscale
if template_image is None:
    raise ValueError("Template image not found")

kp_template, desc_template = sift.detectAndCompute(template_image, None)

# Initialize the camera
video = cv2.VideoCapture(0)  # '0' is the default value for the primary camera

# Set the resolution
desired_width = 640
desired_height = 480
video.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Known dimensions of the object (letter-size paper in inches)
object_height = 250  # letter size paper in mm

# Focal length (fy from the camera matrix since we are using the height)
focal_length = camera_matrix[1, 1]

# Counter for the frames
frame_counter = 0
# Specify the interval for frame processing
frame_interval = 5  # Process every 5 frames

Drive = 90
Angle = 90

# Main loop for video processing
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Increment the frame counter
    frame_counter += 1    
    # Process the frame when frame_counter reaches the interval
    if frame_counter == frame_interval:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, desc_frame = sift.detectAndCompute(gray_frame, None)

        # Feature matching (FLANN based matcher could also be used)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc_template, desc_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        # Homography estimation
        if len(matches) > 3:
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            
            if M is not None:
                h, w = template_image.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                                
                # Height of the object in pixels
                object_height_pixels = np.linalg.norm(dst[0] - dst[1])
                
                # Distance estimation
                distance = (object_height * focal_length) / object_height_pixels
                print(f"Estimated distance: {distance:.2f} mm; ")
                # Calculate the position and distance to midline after finding the object
                position = left_or_right(frame.shape[1], dst)
                midline_distance = distance_to_midline(frame.shape[1], dst)
    
                print(f"Object is on the: {position} side; ")
                print(f"Horizontal distance to midline: {midline_distance:.2f} pixels")
                # Speed setting
                if distance < 250 or distance > 2000:
                    Drive = 90
                    print("Stop")               
                else:
                    Drive = 100
                    print("Forward")
                # Steering setting
                if str(position) == 'Right' and midline_distance > 100:
                    Angle = 135
                    print("Right")
                elif str(position) == 'Left' and midline_distance > 100:
                    Angle = 45
                    print("Left")
                else:
                    Angle= 90
                # Send speed and steering to RC
                send_command(Angle, Drive)
                
        # Reset the counter
        frame_counter = 0


# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()
