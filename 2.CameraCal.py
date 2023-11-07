import numpy as np
import cv2
import glob

# Termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Square size in millimeters
square_size = 23.0  # or the actual size of your squares in millimeters

objp = np.zeros((7*10, 3), np.float32)
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * square_size


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Define the pattern to match .png files in the current directory
images_path = "./*.png"

# Use glob to get all the images with .png extension in the current directory
all_images = glob.glob(images_path)

# Exclude "redhawk.png" from the list of images
images = [img for img in all_images if "redhawk.png" not in img]

# Check if images list is empty
if not images:
    print("No images found. Check the file path and extension.")
else:
    print(f"Found {len(images)} images.")

# Loop through the images
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load image: {fname}")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (10,7), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (10,7), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

cv2.destroyAllWindows()

# Add a check before calling calibrateCamera
if objpoints and imgpoints:
    # Proceed with calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera calibrated successfully.")
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    # Save the calibration data
    np.savez('calibration_data.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("Calibration data saved as 'calibration_data.npz'.")

else:
    print("objpoints and imgpoints are empty. Calibration cannot be performed.")
