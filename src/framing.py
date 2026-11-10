"""
Per-dataset framing policy
==========================

Some datasets in the training pool come pre-cropped around the iris
(MOBIUS, polyu_zoomed, ubiris_v2_with_pupils_zoomed, tayed_eyes). Others are
uncropped full-frame captures where the iris occupies a small fraction of the
image (ubiris_v2_with_pupils: the round-trip output of
``tools/prepare/ubiris_v2_unzoom_pupils.py``, polyu uncropped 640×480).

If we feed both into the same model without per-sample framing, the uncropped
ones drop to ~7% iris-fill after the global resize to 160×160 and the model
either ignores them or learns to ALWAYS expect the eye centered. We want
neither.

This module defines:

* ``FRAMING_POLICY``: dataset_name → policy.
* ``random_iris_framing(img, mask, rng)``: train-time: random center offset +
  random zoom that still contains the whole iris+pupil bbox.
* ``tight_iris_framing(img, mask)``: val/test deterministic: centered crop
  sized so the iris fills a target fraction of the frame. No randomness.
* ``apply_framing(img, mask, policy, augment, rng)``: dispatch entry point.

Policy values
-------------
``"random_iris_framing"``
    Train: random framing. Val/test: tight deterministic crop.
``"none"``
    Pass through unchanged (image is already framed).

If a dataset is not listed in ``FRAMING_POLICY`` it defaults to ``"none"``:
matching the historical behavior so this module is backward-compatible.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# Datasets WHITELISTED for random framing. Anything not here is left untouched.
# Keep this small and explicit so it's obvious from the diff which datasets the
# augmentation can possibly fire on.
FRAMING_POLICY: dict[str, str] = {
    "ubiris_v2_with_pupils": "random_iris_framing",
}


# ---------- core geometry (re-used by both random and tight variants) ----------

def _iris_pupil_bbox(mask_classes: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (y0, x0, y1, x1) bbox of class-1 (iris) ∪ class-2 (pupil)."""
    fg = (mask_classes == 1) | (mask_classes == 2)
    if not fg.any():
        return None
    ys, xs = np.where(fg)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _crop_box(
    mask_classes: np.ndarray,
    target_iris_pct: float,
    jitter_pct: float,
    zoom_jitter: float,
    rng: Optional[np.random.Generator],
    enable_jitter: bool,
) -> Optional[Tuple[int, int, int, int]]:
    """Square crop around the iris+pupil bbox.

    Crop side = bbox_long / target_iris_pct (so iris fills target_iris_pct).
    When ``enable_jitter``, multiply the side by ``1 + U(-zj, +zj)`` and shift
    the center by ``U(-jp, +jp) * crop_side``. Result is clamped to image
    bounds so the crop is always fully inside the frame.
    """
    bb = _iris_pupil_bbox(mask_classes)
    if bb is None:
        return None
    H, W = mask_classes.shape
    y0, x0, y1, x1 = bb
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0
    bbox_long = max(y1 - y0, x1 - x0)
    if bbox_long <= 0:
        return None

    crop_side = bbox_long / max(target_iris_pct, 1e-3)
    if enable_jitter and rng is not None:
        crop_side *= 1.0 + rng.uniform(-zoom_jitter, zoom_jitter)
        jx = rng.uniform(-jitter_pct, jitter_pct) * crop_side
        jy = rng.uniform(-jitter_pct, jitter_pct) * crop_side
        cx += jx
        cy += jy

    # Minimum side: the bbox itself must fit (otherwise we'd cut the iris).
    crop_side = max(crop_side, float(bbox_long))
    # Maximum side: the frame.
    crop_side = min(crop_side, float(min(H, W)))
    half = crop_side / 2.0

    # Clamp center so the crop stays inside the frame.
    cx = float(np.clip(cx, half, W - half))
    cy = float(np.clip(cy, half, H - half))

    return (int(round(cy - half)), int(round(cx - half)),
            int(round(cy + half)), int(round(cx + half)))


# ---------- public variants ----------

def random_iris_framing(
    img: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    target_iris_pct: float = 0.35,
    jitter_pct: float = 0.15,
    zoom_jitter: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """TRAIN-time: random square crop that frames the iris with random scale
    and offset. Guarantees the entire iris+pupil bbox stays in the crop."""
    box = _crop_box(mask, target_iris_pct, jitter_pct, zoom_jitter, rng,
                    enable_jitter=True)
    if box is None:
        return img, mask
    y0, x0, y1, x1 = box
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def tight_iris_framing(
    img: np.ndarray,
    mask: np.ndarray,
    target_iris_pct: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """VAL/TEST: deterministic centered crop sized so the iris fills
    ``target_iris_pct`` of the frame. No randomness."""
    box = _crop_box(mask, target_iris_pct, 0.0, 0.0, None, enable_jitter=False)
    if box is None:
        return img, mask
    y0, x0, y1, x1 = box
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


# ---------- dispatcher used by EyeDataset ----------

def apply_framing(
    img: np.ndarray,
    mask: np.ndarray,
    dataset_name: str,
    augment: bool,
    rng: np.random.Generator,
    target_iris_pct: float = 0.35,
    jitter_pct: float = 0.15,
    zoom_jitter: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the framing policy for ``dataset_name``. No-op if not listed."""
    policy = FRAMING_POLICY.get(dataset_name, "none")
    if policy == "none":
        return img, mask
    if policy == "random_iris_framing":
        if augment:
            return random_iris_framing(img, mask, rng,
                                       target_iris_pct, jitter_pct, zoom_jitter)
        return tight_iris_framing(img, mask, target_iris_pct)
    raise ValueError(f"Unknown framing policy '{policy}' for {dataset_name!r}")
