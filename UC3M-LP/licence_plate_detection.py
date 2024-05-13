import os
from os.path import join
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

import util
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('best.pt')

# Folder containing images (adjust here)
image_folder = "C:/Users/adamb/Desktop/Data/UC3M-LP-yolo/LP/images/val"

# List image files in the folder
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(join(image_folder, f))]
image_file = "00014.jpg"
vehicles = [2, 3, 5, 7]

frame = cv2.imread(join(image_folder, image_file))
frame_resized = cv2.resize(frame, (608, 456))  # Resize to 608x456

results[0] = {}

# Detect vehicles
detections = coco_model(frame_resized)[0]
detections_ = []
for detection in detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection
    if int(class_id) in vehicles:
        detections_.append([x1, y1, x2, y2, score])

print("Detections: ", detections_)

# Check if there are detections before tracking
if detections_:
    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))
    print(track_ids)
    # Detect license plates
    license_plates = license_plate_detector(frame_resized)[0]
    print("Licence plates: ", license_plates.boxes.data.tolist())

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        # Assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
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
            if license_plate_text is not None:
                results[0][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
            # Draw rectangle around license plate
            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Convert BGR to RGB
frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

# Show the image with rectangles drawn around license plates
plt.imshow(frame_rgb)
plt.axis('off')  # Turn off axis labels
plt.show()

# Write results
write_csv(results, './test.csv')
