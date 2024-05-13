from os.path import join
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from util import read_license_plate, write_csv

# consts
FRAME_HEIGHT = 456
FRAME_WIDTH = 608


results = []

# Load models
license_plate_detector = YOLO('best100.pt')

# Folder containing images (adjust here)
image_folder = "C:/Users/adamb/Desktop/Data/UC3M-LP-yolo/LP/images/val"

# input file
image_file = "00000.jpg"

frame = cv2.imread(join(image_folder, image_file))
original_h, original_w = frame.shape[:2]
frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))  # Resize to 608x456

# Specify search frame as whole image
search_frame = [[0, 0, frame_resized.shape[1], frame_resized.shape[0], 1.0]]

# Detect license plates
license_plates = license_plate_detector(frame_resized)[0]


for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, _ = license_plate
    # Crop license plate
    license_plate_crop = frame_resized[int(y1):int(y2), int(x1): int(x2), :]

    # Process license plate
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

    # Read license plate number
    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
    print(license_plate_text)

    # Append results with necessary fields
    results.append({'license_plate': {'bbox': [x1*original_w/FRAME_WIDTH, y1*original_h/FRAME_HEIGHT,
                                               x2*original_w/FRAME_WIDTH, y2*original_h/FRAME_HEIGHT],
                                      'text': license_plate_text,
                                      'bbox_score': score,
                                      'text_score': license_plate_text_score}})

    # Draw rectangle around license plate
    cv2.rectangle(frame, (int(x1*original_w/FRAME_WIDTH), int(y1*original_h/FRAME_HEIGHT)),
                  (int(x2*original_w/FRAME_WIDTH), int(y2*original_h/FRAME_HEIGHT)), (0, 255, 0), 2)

# Show the image with rectangles drawn around license plates
plt.imshow(frame)
plt.axis('off')
plt.show()

results_dict = {}
for idx, result in enumerate(results):
    results_dict[idx] = result

# Write results
write_csv(results_dict, './test.csv')
