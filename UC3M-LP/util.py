import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'licence_plate_bbox', 'licence_plate_bbox_score', 'licence_number',
                                                'licence_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'licence_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['licence_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['licence_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['licence_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['licence_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['licence_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['licence_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['licence_plate']['text'],
                                                            results[frame_nmr][car_id]['licence_plate']['text_score'])
                            )
        f.close()


def licence_complies_format(text):
    """
    Check if the licence plate text complies with the required format.

    Args:
        text (str): licence plate text.

    Returns:
        bool: True if the licence plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_licence(text):
    """
    Format the licence plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): licence plate text.

    Returns:
        str: Formatted licence plate text.
    """
    #licence_plate_ = ''
    #mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
    #           2: dict_char_to_int, 3: dict_char_to_int}
    #for j in [0, 1, 2, 3, 4, 5, 6]:
    #    if text[j] in mapping[j].keys():
    #        licence_plate_ += mapping[j][text[j]]
    #    else:
    #        licence_plate_ += text[j]

    #return licence_plate_


def read_licence_plate(licence_plate_crop):
    """
    Read the licence plate text from the given cropped image.

    Args:
        licence_plate_crop (PIL.Image.Image): Cropped image containing the licence plate.

    Returns:
        tuple: Tuple containing the formatted licence plate text and its confidence score.
    """

    detections = reader.readtext(licence_plate_crop)
    print(detections)
    sorted_detections = sorted(detections, key=lambda x: x[0][0], reverse=False)
    print(sorted_detections)
    merged_text = ""
    score = 0
    for detection in sorted_detections:
        bbox, text, score = detection

        # text = text.upper().replace(' ', '')
        print("text:", text)
        merged_text = merged_text + text
        # if licence_complies_format(text):
        #    return format_licence(text), score
        # new_text = format_licence(text)
        print("merged:", merged_text)
    return merged_text, score

    # return None, None


def get_car(licence_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the licence plate coordinates.

    Args:
        licence_plate (tuple): Tuple containing the coordinates of the licence plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = licence_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1



def find_overlapping_bboxes(bboxes, threshold=0.9):
    """Funkcja znajdująca zestawy bounding boxów, których obszary pokrywają się w minimum 80%."""
    overlapping_bboxes = []

    for i, bbox1 in enumerate(bboxes):
        for j, bbox2 in enumerate(bboxes[i+1:], start=i+1):
            cover = isCovered(bbox1, bbox2)
            if cover :
                overlapping_bboxes.append((bbox1, bbox2))

    return overlapping_bboxes


def isCovered(bbox1, bbox2):
    x1_1, y1_1, x1_2, y1_2, _, _ = bbox1
    x2_1, y2_1, x2_2, y2_2, _, _ = bbox2

    if (abs(x1_1 - x2_1) <= ((x1_2 - x1_1) / 2)) and (abs(y1_1 - y2_1) <= ((y1_2 - y1_1) / 2)):
        return True
    return False
