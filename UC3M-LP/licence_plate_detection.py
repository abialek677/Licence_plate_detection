import os
from os.path import join
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

import util
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv

# consts
FRAME_HEIGHT = 456
FRAME_WIDTH = 608

results = []

mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('best.pt')

# Folder containing images (adjust here)
image_folder = "C:/Users/adamb/Desktop/Data/UC3M-LP-yolo/LP/images/val"

# List image files in the folder
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(join(image_folder, f))]
image_file = "00617.jpg"
vehicles = [2, 3, 5, 7]

frame = cv2.imread(join(image_folder, image_file))
frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))  # Resize to 608x456
original_h, original_w = frame.shape[:2]


# Detect vehicles
detections = [[0, 0, frame_resized.shape[1], frame_resized.shape[0], 1.0]]

# Check if there are detections before tracking
    # Track vehicles
track_ids = np.asarray(detections)
print(track_ids)
# Detect license plates
license_plates = license_plate_detector(frame_resized)[0]
print("Licence plates: ", license_plates.boxes.data.tolist())


for i, license_plate in enumerate(license_plates.boxes.data.tolist()):
    x1, y1, x2, y2, score, class_id = license_plate
    # Assign license plate to car
    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, detections)
    print("Car id:", car_id)
    if car_id != -1:
        # Crop license plate
        license_plate_crop = frame_resized[int(y1):int(y2), int(x1): int(x2), :]

        # Process license plate
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

        # Read license plate number
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
        print(license_plate_text)

        # Append results with necessary fields
        results.append({'license_plate': {'bbox': [x1*original_w/FRAME_WIDTH, y1*original_h/FRAME_HEIGHT, x2*original_w/FRAME_WIDTH, y2*original_h/FRAME_HEIGHT],
                                          'text': license_plate_text,
                                          'bbox_score': score,
                                          'text_score': license_plate_text_score}})

        # Draw rectangle around license plate
        cv2.rectangle(frame, (int(x1*original_w/FRAME_WIDTH), int(y1*original_h/FRAME_HEIGHT)), (int(x2*original_w/FRAME_WIDTH), int(y2*original_h/FRAME_HEIGHT)), (0, 255, 0), 2)


# Convert BGR to RGB
frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

# Show the image with rectangles drawn around license plates
plt.imshow(frame)
plt.axis('off')  # Turn off axis labels
plt.show()

results_dict = {}
for idx, result in enumerate(results):
    results_dict[idx] = result

# Write results
write_csv(results_dict, './test.csv')


