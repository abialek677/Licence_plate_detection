from os import listdir, makedirs
from os.path import join, exists
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from util import read_licence_plate, write_csv, find_overlapping_bboxes

# Constants
FRAME_HEIGHT = 456
FRAME_WIDTH = 608
DETECTED_FOLDER = "detected_val"

# Load models
licence_plate_detector = YOLO('best100.pt')
licence_plate_recognition = YOLO('best_letters_312.pt')

# Folder containing images
image_folder = "C:/Users/Magda/Desktop/studia/sem4/sztuczna_inteligencja/UC3M-LP-yolo/LP/images/val"

# Create 'detected' subfolder if it doesn't exist
if not exists(DETECTED_FOLDER):
    makedirs(DETECTED_FOLDER)

# Process each image in the folder
for image_file in listdir(image_folder):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        # Read the image
        frame = cv2.imread(join(image_folder, image_file))
        original_h, original_w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))  # Resize to 608x456

        # Detect licence plates
        licence_plates = licence_plate_detector(frame_resized)[0]

        for licence_plate in licence_plates.boxes.data.tolist():
            licence_plate_text = ""
            x1, y1, x2, y2, score, _ = licence_plate
            # Crop licence plate
            licence_plate_crop = frame_resized[int(y1):int(y2), int(x1): int(x2), :]

            # Process licence plate
            licence_plate_crop_gray = cv2.cvtColor(licence_plate_crop, cv2.COLOR_BGR2GRAY)

            # Read licence plate number
            licence_plate_detections = licence_plate_recognition(licence_plate_crop)[0]

            # Convert detections to list of bounding boxes with scores
            bboxes = licence_plate_detections.boxes.data.tolist()

            overlapping_bboxes = find_overlapping_bboxes(bboxes)
            # Group overlapping bounding boxes
            grouped_bboxes = []
            while overlapping_bboxes:
                bbox1, bbox2 = overlapping_bboxes.pop(0)
                group = [bbox1, bbox2]
                i = 0
                while i < len(overlapping_bboxes):
                    bbox1, bbox2 = overlapping_bboxes[i]
                    if any(bbox in group for bbox in (bbox1, bbox2)):
                        group.extend([bbox1, bbox2])
                        overlapping_bboxes.pop(i)
                    else:
                        i += 1
                grouped_bboxes.append(group)

            # Choose the best bounding box from each group
            chosen_detections = []
            for group in grouped_bboxes:
                best_bbox = max(group, key=lambda x: x[4])  # Choose bbox with highest score
                chosen_detections.append(best_bbox)

            # Add non-overlapping bounding boxes to chosen detections
            for bbox in bboxes:
                if bbox not in sum(grouped_bboxes, []):  # Check if bbox is not in any group
                    chosen_detections.append(bbox)

            sorted_detections = sorted(chosen_detections, key=lambda x: x[0])

            for detection in sorted_detections:
                x1_d, y1_d, x2_d, y2_d, score_d, class_id = detection
                licence_plate_text += licence_plate_recognition.names[class_id]

            # Draw rectangle around licence plate
            cv2.rectangle(frame, (int(x1*original_w/FRAME_WIDTH), int(y1*original_h/FRAME_HEIGHT)),
                          (int(x2*original_w/FRAME_WIDTH), int(y2*original_h/FRAME_HEIGHT)), (0, 255, 0), 2)

            # Put a text above the licence plate
            cv2.putText(frame, str(licence_plate_text),
                        (int(x1 * original_w / FRAME_WIDTH), int(y1 * original_h / FRAME_HEIGHT) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the processed image with detected licence plates
        cv2.imwrite(join(DETECTED_FOLDER, image_file), frame)

print("Processing complete.")
