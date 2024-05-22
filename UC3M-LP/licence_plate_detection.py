from os import listdir, makedirs, path, environ
from os.path import join, exists
import cv2
from ultralytics import YOLO
from scripts.utils import find_overlapping_bboxes
import Levenshtein as lev
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Constants
FRAME_HEIGHT = 456
FRAME_WIDTH = 608
DETECTED_FOLDER = "detected_val"

# Load models
licence_plate_detector = YOLO('models/licence_100.pt')
licence_plate_recognition = YOLO('models/letters_best_312.pt')

# checking the work of the ocr model
evaluateFlag = True


# Folder containing images
if evaluateFlag:
    image_folder = "evaluate/cars_test"
else:
    image_folder = "cars"

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

            if evaluateFlag:
                filename, ext = path.splitext(image_file)
                o_path = f"evaluate/model_output_yolo/{filename}.txt"
                with open(o_path, 'w') as f:
                    f.write(licence_plate_text)
                o_path_tesseract = f"evaluate/tesseract_output/{filename}.txt"
                with open(o_path_tesseract, 'w') as f:
                    f.write(pytesseract.image_to_string(licence_plate_crop_gray, config='--oem 0 --psm 6'))

            x1_scaled = int(x1 * original_w / FRAME_WIDTH)
            y1_scaled = int(y1 * original_h / FRAME_HEIGHT)
            x2_scaled = int(x2 * original_w / FRAME_WIDTH)
            y2_scaled = int(y2 * original_h / FRAME_HEIGHT)

            # draw outline
            cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 0, 255), 2)

            # Put the text above the licence plate with white background and black outline
            (text_width, text_height), baseline = cv2.getTextSize(licence_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1_scaled, y1_scaled - text_height - 10),
                          (x1_scaled + text_width, y1_scaled), (255, 255, 255), -1)
            cv2.putText(frame, str(licence_plate_text), (x1_scaled, y1_scaled - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Save the processed image with detected licence plates
        if not evaluateFlag:
            cv2.imwrite(join(DETECTED_FOLDER, image_file), frame)

print("Processing complete.")

if evaluateFlag:
    output_path = "evaluate/model_output_yolo"
    tagged_data_path = "evaluate/test_eval"
    tesseract_path = "evaluate/tesseract_output"

    output_files = [path.join(output_path, file) for file in listdir(output_path)]
    tagged_data_files = [path.join(tagged_data_path, file) for file in listdir(tagged_data_path)]
    tesseract_files = [path.join(tesseract_path, file) for file in listdir(tesseract_path)]
    model_evaluation = 0
    tesseract_evaluation = 0
    for output_file, tagged_data_file, tesseract_file in zip(output_files, tagged_data_files, tesseract_files):
        # Load text from files
        with open(output_file, 'r') as f:
            output_text = f.read()
        with open(tagged_data_file, 'r') as f:
            tagged_data_text = f.read()
        with open(tesseract_file, 'r') as f:
            tesseract_text = f.read()

        distance_model = lev.distance(output_text, tagged_data_text)
        distance_tesseract = lev.distance(tesseract_text, tagged_data_text)

        normalized_distance_model = max(distance_model / len(tagged_data_text), 0)
        model_evaluation += normalized_distance_model

        normalized_distance_tesseract = max(distance_tesseract / len(tagged_data_text), 0)
        tesseract_evaluation += normalized_distance_tesseract

    model_evaluation /= len(output_files)
    model_evaluation = 1 - model_evaluation
    tesseract_evaluation /= len(output_files)
    tesseract_evaluation = 1 - tesseract_evaluation

    print(f"\n\nModel evaluation: {model_evaluation}")
    print(f"Tesseract evaluation: {tesseract_evaluation}")
