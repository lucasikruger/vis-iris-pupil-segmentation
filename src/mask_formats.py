"""
Mask Format Converters
=======================
Each converter takes a BGR image (as read by cv2) and returns an int64
array with class indices:
    0 = background / sclera
    1 = iris
    2 = pupil

All datasets in this project are stored in mobius-format (RGB color-coded),
so all converters apply the same logic:
    Red   (255, 0,   0)   → class 0 (background)
    Green (0,   255, 0)   → class 1 (iris)
    Blue  (0,   0,   255) → class 2 (pupil)
"""

import cv2
import numpy as np
from typing import Callable, Dict


_CONVERTERS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}


def register(name: str):
    def deco(fn):
        _CONVERTERS[name.lower()] = fn
        return fn
    return deco


def get_converter(name: str) -> Callable[[np.ndarray], np.ndarray]:
    key = name.lower()
    if key not in _CONVERTERS:
        raise KeyError(f"Unknown mask format '{name}'. Registered: {list(_CONVERTERS.keys())}")
    return _CONVERTERS[key]


def _rgb_to_classes(mask_bgr: np.ndarray) -> np.ndarray:
    """Shared logic: R=bg(0), G=iris(1), B=pupil(2).

    Convention: pupil pixels = BLUE channel is set (regardless of G).
    This correctly handles two annotation styles:
      - MOBIUS / PolyU: pupil is pure blue (0, 0, 255): iris is a ring with hole
      - UBIRIS-style: pupil is CYAN (0, 255, 255): painted on top of the iris green
    Both decode to class 2 (pupil) because BLUE is set. iris class 1 is only
    set when GREEN is on AND BLUE is off, so the cyan-pupil case ends up as
    pupil, not iris.
    """
    rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    g_on = rgb[..., 1] > 127
    b_on = rgb[..., 2] > 127
    out = np.zeros(rgb.shape[:2], dtype=np.int64)
    out[g_on & ~b_on] = 1  # iris  (green only)
    out[b_on]         = 2  # pupil (blue set. takes priority over iris)
    return out


@register("mobius")
@register("mobius_3c")
@register("mobius_rgb")
@register("iris_pupil_eye_cls")
@register("unity_eyes_3c")
@register("tayed_3c")
@register("periorbital_3c")
@register("ubipr_3c")
def mobius_format(mask_bgr: np.ndarray) -> np.ndarray:
    return _rgb_to_classes(mask_bgr)


@register("mobius_4c")
def mobius_4c_format(mask_bgr: np.ndarray) -> np.ndarray:
    """4-class MOBIUS-native: bg=0, sclera=1, iris=2, pupil=3.

    Convention (BGR uint8, cv2.imread output):
      - R-only pixels (BGR = 0,0,255)   → class 1 (sclera)
      - G-only pixels (BGR = 0,255,0)   → class 2 (iris)
      - B-set pixels  (BGR = 255,*,*)   → class 3 (pupil. takes priority)

    This is used by the sclera annotator (Stage 1): trained ONLY on MOBIUS,
    later used to derive sclera masks for the other datasets that only have
    iris+pupil annotations.
    """
    rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    r_on = rgb[..., 0] > 127
    g_on = rgb[..., 1] > 127
    b_on = rgb[..., 2] > 127
    out = np.zeros(rgb.shape[:2], dtype=np.int64)
    out[r_on & ~g_on & ~b_on] = 1   # sclera (red only)
    out[g_on & ~b_on]         = 2   # iris   (green, no blue)
    out[b_on]                 = 3   # pupil  (blue set)
    return out


@register("ubiris_v2_iris_only")
def ubiris_v2_format(mask_bgr: np.ndarray) -> np.ndarray:
    """UBIRIS v2: iris only (green channel), no pupil annotation."""
    rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    out = np.zeros(rgb.shape[:2], dtype=np.int64)
    out[(rgb[..., 0] == 0) & (rgb[..., 1] == 255) & (rgb[..., 2] == 0)] = 1  # iris
    return out
