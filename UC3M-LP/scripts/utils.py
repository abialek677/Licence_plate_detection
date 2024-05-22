import os


def poly2bbox(poly_coord):
    x_coords = [coord[0] for coord in poly_coord]
    y_coords = [coord[1] for coord in poly_coord]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    return [[x_min, y_min], [x_max, y_max]]


def create_txt_file(input_directory):
    # Read files in lp_directory
    train_files = os.listdir(os.path.join(input_directory, 'images', 'train'))
    val_files = os.listdir(os.path.join(input_directory, 'images', 'val'))
    train_files.sort()
    val_files.sort()

    # Save train and val files in train.txt and val.txt
    with open(os.path.join(input_directory, 'train.txt'), 'w') as f:
        for filename in train_files:
            f.write(filename.split('.')[0] + '\n')
    
    with open(os.path.join(input_directory, 'val.txt'), 'w') as f:
        for filename in val_files:
            f.write(filename.split('.')[0] + '\n')

def find_overlapping_bboxes(bboxes):
    """Funkcja znajdująca zestawy bounding boxów, których obszary pokrywają się w minimum 80%."""
    overlapping_bboxes = []

    for i, bbox1 in enumerate(bboxes):
        for j, bbox2 in enumerate(bboxes[i+1:], start=i+1):
            cover = isCovered(bbox1, bbox2)
            if cover:
                overlapping_bboxes.append((bbox1, bbox2))

    return overlapping_bboxes


def isCovered(bbox1, bbox2):
    x1_1, y1_1, x1_2, y1_2, _, _ = bbox1
    x2_1, y2_1, x2_2, y2_2, _, _ = bbox2

    if (abs(x1_1 - x2_1) <= ((x1_2 - x1_1) / 2)) and (abs(y1_1 - y2_1) <= ((y1_2 - y1_1) / 2)):
        return True
    return False
