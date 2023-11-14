import numpy as np
import cv2
import threading

# Function to determine if the object is on the left or right side of the screen
def left_or_right(frame_width, dst):
    centroid_x = np.mean(dst[:, 0, 0])
    return "Right" if centroid_x > frame_width / 2 else "Left"

# Function to calculate the horizontal distance to the midline
def distance_to_midline(frame_width, dst):
    centroid_x = np.mean(dst[:, 0, 0])
    return np.abs(centroid_x - frame_width / 2)

# Class for handling video capture in a separate thread
class CameraThread:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        (self.ret, frame) = self.capture.read()
        self.frame = cv2.UMat(frame)  # Storing the frame as UMat for GPU acc
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            (self.ret, frame) = self.capture.read()
            if self.ret:
                self.frame = cv2.UMat(frame)  # Update the frame as UMat

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.capture.release()


# Load previously saved data from the .npz file
with np.load('calibration_data.npz') as X:
    camera_matrix, dist_coeffs = [X[i] for i in ('mtx', 'dist')]

# Focal length (fy from the camera matrix since we are using the height)
focal_length = camera_matrix[1, 1]

# Known dimensions of the object (in mm)
object_height = 210  # Height of the object (letter-size paper) in mm

# Initialize the ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Load the template image
template_image = cv2.UMat(cv2.imread('Redhawk.jpg'))
if template_image is None:
    raise ValueError("Template image not found")

kp_template, desc_template = orb.detectAndCompute(template_image, None)

# Initialize the camera thread
camera_thread = CameraThread().start()

frame_counter = 0
frame_interval = 60  # Process every 60 frame
last_line_coords = None

# Main loop for video processing
while True:
    frame = camera_thread.read()
    if frame is None:
        break

    frame_counter += 1

    if frame_counter % frame_interval == 0:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)

        # Feature matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc_template, desc_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        # Homography estimation
        if len(matches) > 1:
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

            if M is not None:
                h, w = template_image.get().shape[:2]   # Get the height and width of the template image
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                last_line_coords = np.int32(dst)

                frame_np = frame.get() if isinstance(frame, cv2.UMat) else frame
                frame_np = cv2.polylines(frame_np, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                frame = cv2.UMat(frame_np) if isinstance(frame, cv2.UMat) else frame_np

                # Height of the object in pixels
                object_height_pixels = np.linalg.norm(dst[0] - dst[1])

                # Distance estimation
                distance = (object_height * focal_length) / object_height_pixels
                print(f"Estimated distance: {distance:.2f} mm; ")

                # Calculate the position and distance to midline
                position = left_or_right(frame_np.shape[1], dst)
                midline_distance = distance_to_midline(frame_np.shape[1], dst)

                print(f"Object is on the: {position} side; ")
                print(f"Horizontal distance to midline: {midline_distance:.2f} pixels")
        # Reset the counter
        frame_counter = 0
    if last_line_coords is not None:
        frame_np = frame.get() if isinstance(frame, cv2.UMat) else frame
        frame_np = cv2.polylines(frame_np, [last_line_coords], True, (0, 255, 0), 3, cv2.LINE_AA)
        frame = cv2.UMat(frame_np) if isinstance(frame, cv2.UMat) else frame_np

    frame_display = frame.get() if isinstance(frame, cv2.UMat) else frame
    cv2.imshow('frame', frame_display)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
camera_thread.stop()
cv2.destroyAllWindows()
