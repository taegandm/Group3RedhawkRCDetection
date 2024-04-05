import numpy as np
import cv2

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

# Capture video stream from the webcam
video = cv2.VideoCapture(0)
# Set the resolution to 1920x1080
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Known dimensions of the object (letter-size paper in inches)
object_height = 250  # letter size paper in mm

# Focal length (fy from the camera matrix since we are using the height)
focal_length = camera_matrix[1, 1]

# Counter for the frames
frame_counter = 0
# Specify the interval for frame processing
frame_interval = 5  # Process every 5 frames
# Define a variable to store the last drawing box coordinates
last_box = None

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

        # FLANN parameters and matcher initialization
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass an empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # Feature matching
        matches = flann.knnMatch(desc_template, desc_frame, k=2)
        # Extract good matches for Homography
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Homography estimation
        if len(good_matches) > 3:
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            
            if M is not None:
                h, w = template_image.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                last_box = dst  # Update last box coordinates
                
                #frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                
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
        
        # Reset the counter
        frame_counter = 0
    
    if last_box is not None:  # Draw the last detected box
        frame = cv2.polylines(frame, [np.int32(last_box)], True, (0, 255, 0), 3, cv2.LINE_AA)
    
    cv2.imshow('ESC or Q for Exit', frame)
    
    # Exit loop if 'q' or 'Q' or 'ESC' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q') or key == ord('Q'):
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()
