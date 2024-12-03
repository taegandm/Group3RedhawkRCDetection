import torch
import cv2
import numpy as np

# Load YOLOv5 model (the small version is faster but less accurate)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is a small model, use 'yolov5m', 'yolov5l', 'yolov5x' for better accuracy

# Optionally, you can specify other models like 'yolov5m', 'yolov5l', or 'yolov5x' based on speed/accuracy tradeoff
# Load an image
img = cv2.imread('your_image_path.jpg')

# Perform detection on the image
results = model(img)

# Print results (bounding boxes, labels, and confidence scores)
results.print()

# Display results
results.show()
