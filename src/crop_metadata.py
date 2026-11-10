"""Recorded crop geometry in original-image pixels. no detector is rerun."""
import json
from pathlib import Path
import cv2

INTERPOLATIONS = {'AREA': cv2.INTER_AREA, 'CUBIC': cv2.INTER_CUBIC,
                  'LINEAR': cv2.INTER_LINEAR, 'LANCZOS4': cv2.INTER_LANCZOS4}


def load_boxes(package_root, source):
    path = Path(package_root) / 'annotations' / f'{source}_boxes.json'
    document = json.loads(path.read_text())
    if document['schema_version'] != 1 or document['coordinate_system'] != 'pixel_xyxy_half_open':
        raise ValueError(f'Unsupported crop coordinates: {path}')
    return document['records']


def relative_path(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Expected a safe relative path: {name}')
    return Path(root) / path


def crop_from_record(image, record, *, box_key='bbox_xyxy', size_key='output_size', interpolation=None):
    if image is None:
        raise ValueError('Original image is missing or unreadable')
    h, w = image.shape[:2]
    if [w, h] != record['source_size']:
        raise ValueError(f'Original dimensions {[w,h]} != recorded {record["source_size"]}')
    box = record.get(box_key)
    if box is None or len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError('An exact integer crop box is required')
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f'Invalid crop box: {box}')
    result = image[y0:y1, x0:x1]
    size = tuple(record[size_key])
    mode = interpolation or record['resize']
    if mode == 'none':
        if [x1-x0, y1-y0] != list(size):
            raise ValueError('A resize is required but not recorded')
        return result.copy()
    return cv2.resize(result, size, interpolation=INTERPOLATIONS[mode])
