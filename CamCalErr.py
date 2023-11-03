import numpy as np
import cv2
import glob

# Termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the chessboard dimensions
# (7x10 inner corners in your case)
objp = np.zeros((7*10, 3), np.float32)
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Define the path to the images
images_path = "D:\\openCV\\*.png"
# Use glob to get all the images with .png extension
images = glob.glob(images_path)

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

    # Calculate and print the re-projection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error
    print("Total re-projection error:", total_error/len(objpoints))

else:
    print("objpoints and imgpoints are empty. Calibration cannot be performed.")
