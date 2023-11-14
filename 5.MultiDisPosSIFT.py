import numpy as np
import cv2
import threading
import queue

# Function to determine if the object is on the left or right side of the screen
def left_or_right(frame_width, dst):
    centroid_x = np.mean(dst[:, 0, 0])
    return "Right" if centroid_x > frame_width / 2 else "Left"

# Function to calculate the horizontal distance to the midline
def distance_to_midline(frame_width, dst):
    centroid_x = np.mean(dst[:, 0, 0])
    return np.abs(centroid_x - frame_width / 2)

# Capture thread function
def capture_thread(video, frame_queue):
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_queue.put(frame)

# Processing thread function
def processing_thread(frame_queue, camera_matrix, dist_coeffs, sift, template_image, object_height, focal_length):
    # Counter for the frames
    frame_counter = 0
    # Specify the interval for frame processing
    frame_interval = 5  # Process every x frames
    
    last_box = None  # Initialize the last box variable

    while True:
        if not frame_queue.empty():
            
            frame = frame_queue.get()
            frame_counter += 1  # Increment frame counter
            
            # Process the frame when frame_counter reaches the interval

            if frame_counter % frame_interval==0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kp_frame, desc_frame = sift.detectAndCompute(gray_frame, None)

                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                matches = bf.match(desc_template, desc_frame)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > 3:
                    src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

                    if M is not None:
                        h, w = template_image.shape
                        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        last_box = dst  # Update last box coordinates

                        frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 5, cv2.LINE_AA)

                        object_height_pixels = np.linalg.norm(dst[0] - dst[1])
                        distance = (object_height * focal_length) / object_height_pixels

                        position = left_or_right(frame.shape[1], dst)
                        midline_distance = distance_to_midline(frame.shape[1], dst)

                        print(f"Estimated distance: {distance:.2f} mm")
                        print(f"Object is on the: {position} side")
                        print(f"Horizontal distance to midline: {midline_distance:.2f} pixels")        
            if last_box is not None:  # Draw the last detected box
                frame = cv2.polylines(frame, [np.int32(last_box)], True, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('SIFTMultiDisPos', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            

# Main script
with np.load('calibration_data.npz') as X:
    camera_matrix, dist_coeffs = [X[i] for i in ('mtx', 'dist')]

sift = cv2.SIFT_create()
template_image = cv2.imread('Redhawk.jpg', 0)
if template_image is None:
    raise ValueError("Template image not found")
kp_template, desc_template = sift.detectAndCompute(template_image, None)

object_height = 210  # in mm
focal_length = camera_matrix[1, 1]

frame_queue = queue.Queue()

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Start capture and processing threads
capture = threading.Thread(target=capture_thread, args=(video, frame_queue))
processing = threading.Thread(target=processing_thread, args=(frame_queue, camera_matrix, dist_coeffs, sift, template_image, object_height, focal_length))

capture.start()
processing.start()

capture.join()
processing.join()

video.release()

cv2.destroyAllWindows()