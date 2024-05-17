from os.path import join
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from util import read_licence_plate, write_csv

# consts
FRAME_HEIGHT = 456
FRAME_WIDTH = 608


results = []

# Load models
licence_plate_detector = YOLO('best100.pt')
licence_plate_recognition = YOLO('best_letters.pt')
#licence_plate_recognition = YOLO('best_letters_312.pt')

# Folder containing images (adjust here)
#image_folder = "C:/Users/adamb/Desktop/Data/UC3M-LP-yolo/LP/images/val"
image_folder = "C:/Users/Magda/Desktop/studia/sem4/sztuczna_inteligencja/UC3M-LP-yolo/LP/images/val"

# input file
image_file = "00000.jpg"

frame = cv2.imread(join(image_folder, image_file))
original_h, original_w = frame.shape[:2]
frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))  # Resize to 608x456

# Specify search frame as whole image
search_frame = [[0, 0, frame_resized.shape[1], frame_resized.shape[0], 1.0]]

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

    # Sorting letter/numbers in order from left to right
    sorted_detections = sorted(licence_plate_detections.boxes.data.tolist(), key=lambda x: x[0])

    for detection in sorted_detections:
        x1_d, y1_d, x2_d, y2_d, score_d, class_id = detection
        licence_plate_text += licence_plate_recognition.names[class_id]

        # Append results with necessary fields
        #results.append({'licence_plate': {'bbox': [x1*original_w/FRAME_WIDTH, y1*original_h/FRAME_HEIGHT,
        #                                           x2*original_w/FRAME_WIDTH, y2*original_h/FRAME_HEIGHT],
        #                                  'text': licence_plate_text,
        #                                  'bbox_score': score}})

    # Draw rectangle around licence plate
    cv2.rectangle(frame, (int(x1*original_w/FRAME_WIDTH), int(y1*original_h/FRAME_HEIGHT)),
                  (int(x2*original_w/FRAME_WIDTH), int(y2*original_h/FRAME_HEIGHT)), (0, 255, 0), 2)

    # Put a text above the licence plate
    cv2.putText(frame, str(licence_plate_text),
                (int(x1 * original_w / FRAME_WIDTH), int(y1 * original_h / FRAME_HEIGHT) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Show the image with rectangles drawn around licence plates and licence plate number above
plt.imshow(frame)
plt.axis('off')
plt.show()

results_dict = {}
for idx, result in enumerate(results):
    results_dict[idx] = result

# Write results
write_csv(results_dict, './test.csv')
