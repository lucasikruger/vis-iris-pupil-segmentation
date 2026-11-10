"""Morphological cleanup of iris and pupil labels (0 background, 1 iris, 2 pupil)."""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


def largest(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.copy()
    return labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def postprocess(mask):
    """Thesis morph variant. An absent pupil remains absent."""
    disc = binary_fill_holes(largest(mask > 0))
    pupil = largest(mask == 2)
    disc = cv2.morphologyEx(disc.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    pupil = cv2.morphologyEx(pupil.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    output = np.zeros_like(mask)
    output[disc] = 1
    output[disc & pupil] = 2
    return output
