import numpy as np
import cv2
import glob

# Load previously saved calibration data
with np.load('calibration_data.npz') as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]

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
    print(f"Found {len(images)} images for undistortion.")

# Starting undistortion process
for i, fname in enumerate(images):
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    # Save the undistorted image
    cv2.imwrite(f"./undistorted_{i}.png", dst)

print("Undistortion completed. Undistorted images saved.")
