import os
import json
import matplotlib.pyplot as plt
import cv2
import argparse
from tqdm import tqdm
import utils
import shutil
import numpy as np


def create_yolo_bbox_string(class_id, bbox, img_width, img_height):
    x_center = (bbox[0][0] + bbox[1][0]) / (2 * img_width)
    y_center = (bbox[0][1] + bbox[1][1]) / (2 * img_height)
    width = (bbox[1][0] - bbox[0][0]) / img_width
    height = (bbox[1][1] - bbox[0][1]) / img_height
    return f'{class_id} {x_center} {y_center} {width} {height}'

def clean_filename(filename):
    # Niedozwolone znaki w nazwach plików w systemie Windows: \ / : * ? " < > |
    # Zamień niedozwolone znaki na podkreślenia
    invalid_chars = r'\/:*?"<>|'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def transform_dataset(input_directory, lp_size, ocr_size):
    train_txt_path = os.path.join(input_directory, 'train.txt')
    test_txt_path = os.path.join(input_directory, 'test.txt')

    # Normalize input directory
    input_directory = os.path.normpath(input_directory)

    last_dir = os.path.basename(os.path.normpath(input_directory)) + '-yolo'
    lp_directory = os.path.join(os.path.dirname(input_directory),
                                last_dir, 'LP')
    ocr_directory = os.path.join(os.path.dirname(input_directory),
                                 last_dir, 'OCR')

    # Create directories if not exist
    # Create directories if not exist with read, write, and execute permissions for the owner
    os.makedirs(os.path.join(lp_directory, 'images', 'train'), exist_ok=True, mode=0o700)
    os.makedirs(os.path.join(lp_directory, 'images', 'val'), exist_ok=True, mode=0o700)
    os.makedirs(os.path.join(lp_directory, 'labels', 'train'), exist_ok=True, mode=0o700)
    os.makedirs(os.path.join(lp_directory, 'labels', 'val'), exist_ok=True, mode=0o700)
    os.makedirs(os.path.join(ocr_directory, 'images', 'train'), exist_ok=True, mode=0o700)
    os.makedirs(os.path.join(ocr_directory, 'images', 'val'), exist_ok=True, mode=0o700)
    os.makedirs(os.path.join(ocr_directory, 'labels', 'train'), exist_ok=True, mode=0o700)
    os.makedirs(os.path.join(ocr_directory, 'labels', 'val'), exist_ok=True, mode=0o700)

    ocr_classes = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for txt_file in [train_txt_path, test_txt_path]:
        split = 'train' if 'train' in txt_file else 'test'
        yolo_split = 'train' if 'train' in txt_file else 'val'

        with open(txt_file, 'r') as f:
            filenames = f.read().splitlines()

        print(f'Processing {split} split')
        for filename in tqdm(filenames):
            # Load image
            img_path = os.path.join(input_directory, split, filename + '.jpg')
            img = cv2.imread(img_path)
            img_height, img_width, _ = img.shape
            # Load JSON label
            json_path = os.path.join(input_directory, split, filename + '.json')
            with open(json_path, 'r') as f:
                data = json.load(f)

            # License Plate Detection Dataset
            for lp_data in data['lps']:
                lp_img = img.copy()
                lp_id = lp_data['lp_id']
                poly_coord = lp_data['poly_coord']

                # Convert polygonal annotation to rectangular bbox
                lp_bbox = utils.poly2bbox(poly_coord)

                # Write license plate image
                lp_output_path = os.path.join(lp_directory, 'images', yolo_split, f'{filename}.jpg')
                # Check if file exists
                if not os.path.isfile(os.path.dirname(lp_output_path)):
                    rescale_factor_lp = lp_size / max(img_height, img_width)
                    # Resize license plate to desired size
                    lp_img_resized = cv2.resize(lp_img, (int(img_width * rescale_factor_lp),
                                                         int(img_height * rescale_factor_lp)))

                    if not os.path.isfile(lp_output_path):
                        cv2.imwrite(lp_output_path, lp_img_resized)
                    else:
                        print("File already exists, skipping...")

                # Write YOLO bbox annotation for license plate
                lp_yolo_path = os.path.join(lp_directory, 'labels', yolo_split, f'{filename}.txt')
                append_write_lp = 'a' if os.path.exists(lp_yolo_path) else 'w'
                with open(lp_yolo_path, append_write_lp) as lp_f:
                    lp_f.write(create_yolo_bbox_string(0, lp_bbox, img_width, img_height) + '\n')

                # Write OCR image
                ocr_output_path = os.path.join(ocr_directory, 'images', yolo_split, f'{filename}_{lp_id}.jpg')
                # Crop lp_img to lp_bbox
                ocr_img = lp_img[lp_bbox[0][1]:lp_bbox[1][1], lp_bbox[0][0]:lp_bbox[1][0]]
                ocr_img_offset_x = lp_bbox[0][0]
                ocr_img_offset_y = lp_bbox[0][1]
                ocr_height, ocr_width, _ = ocr_img.shape
                # Resize OCR image to desired size
                rescale_factor_ocr = ocr_size / max(ocr_height, ocr_width)
                ocr_img_resized = cv2.resize(ocr_img, (int(ocr_width * rescale_factor_ocr), int(ocr_height * rescale_factor_ocr)))

                file_name = os.path.basename(ocr_output_path)
                file_name_clear = clean_filename(file_name)
                ocr_output_path_clear = os.path.join(os.path.dirname(ocr_output_path), file_name_clear)
                if not os.path.isfile(ocr_output_path_clear):
                    cv2.imwrite(ocr_output_path_clear, ocr_img_resized)
                else:
                    print("File already exists, skipping...")

                # OCR Detection Dataset
                for char_data in lp_data['characters']:
                    char_id = char_data['char_id']
                    bbox = char_data['bbox_coord']
                    bbox = [[bbox[0][0] - ocr_img_offset_x, bbox[0][1] - ocr_img_offset_y],
                            [bbox[1][0] - ocr_img_offset_x, bbox[1][1] - ocr_img_offset_y]]
                    class_id = ocr_classes.index(char_id)

                    # Write YOLO bbox annotation for character
                    ocr_yolo_path = os.path.join(ocr_directory, 'labels', yolo_split, f'{filename}_{lp_id}.txt')
                    file_name_yolo = os.path.basename(ocr_yolo_path)
                    file_name_yolo_clear = clean_filename(file_name_yolo)
                    ocr_yolo_path_clean = os.path.join(os.path.dirname(ocr_yolo_path), file_name_yolo_clear)
                    append_write_ocr = 'a' if os.path.exists(ocr_yolo_path_clean) else 'w'
                    with open(ocr_yolo_path_clean, append_write_ocr) as ocr_f:
                        ocr_f.write(create_yolo_bbox_string(class_id, bbox, ocr_width, ocr_height) + '\n')

    utils.create_txt_file(lp_directory)
    utils.create_txt_file(ocr_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_directory', type=str, help='Path to input dataset')
    parser.add_argument('lp_size', type=int, help='YOLO input size for LP detection')
    parser.add_argument('ocr_size', type=int, help='YOLO input size for OCR detection')
    args = parser.parse_args()

    transform_dataset(args.input_directory, args.lp_size, args.ocr_size)