"""
Eye Segmentation Dataset
=========================
PyTorch Dataset that reads from a CSV file with columns:
    rel_image_path  - image path relative to data_root
    rel_mask_path   - mask path relative to data_root
    dataset_format  - mask encoding (see mask_formats.py)
    split           - train / val / test (used for filtering externally)
    dataset         - dataset name (mobius, irispupileye, tayed_eyes, unity_eyes)

Preprocessing (CLAHE + gamma correction) is applied to ALL splits.
Augmentation (flip, translation, blur, lines, starburst, brightness, noise)
is applied only when augment=True (training mode).

See src/augmentations.py for the full pipeline.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augmentations import apply_augmentations_v3, preprocess
from .framing import FRAMING_POLICY, apply_framing
from .mask_formats import get_converter


def _mask_bbox(mask_classes: np.ndarray, target_classes=(1, 2)) -> Tuple[int, int, int, int] | None:
    """Return (y0, x0, y1, x1) bbox of pixels matching any class in target_classes."""
    fg = np.isin(mask_classes, target_classes)
    if not fg.any():
        return None
    ys, xs = np.where(fg)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _online_crop_box(
    mask_classes: np.ndarray,
    target_iris_pct: float,
    jitter_pct: float,
    zoom_jitter: float,
    rng: np.random.Generator,
    enable_jitter: bool,
) -> Tuple[int, int, int, int] | None:
    """Compute a square crop box around the iris bbox.

    The crop is sized so the longer side of the iris+pupil bbox fills
    `target_iris_pct` of the crop. When `enable_jitter`, the centre is
    offset by up to `jitter_pct * crop_side` and the crop_side multiplied
    by a random factor in [1 - zoom_jitter, 1 + zoom_jitter]. Result is
    clamped to image bounds.

    Returns None if there's no iris/pupil in the mask (caller can fall
    back to full-image resize).
    """
    bb = _mask_bbox(mask_classes, target_classes=(1, 2))
    if bb is None:
        return None
    H, W = mask_classes.shape
    y0, x0, y1, x1 = bb
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0
    bbox_long = max(y1 - y0, x1 - x0)

    # Default crop side = bbox_long / target_iris_pct (square)
    crop_side = bbox_long / max(target_iris_pct, 1e-3)

    if enable_jitter:
        crop_side *= 1.0 + rng.uniform(-zoom_jitter, zoom_jitter)
        # Jitter centre, but keep mask center within range. cap at jitter_pct of crop_side
        jx = rng.uniform(-jitter_pct, jitter_pct) * crop_side
        jy = rng.uniform(-jitter_pct, jitter_pct) * crop_side
        cx += jx
        cy += jy

    # Clamp crop to image bounds (keep square. shrink if needed)
    crop_side = min(crop_side, min(H, W))
    half = crop_side / 2.0
    cx_min = half
    cy_min = half
    cx_max = W - half
    cy_max = H - half
    cx = float(np.clip(cx, cx_min, cx_max))
    cy = float(np.clip(cy, cy_min, cy_max))

    y_lo = int(round(cy - half))
    x_lo = int(round(cx - half))
    y_hi = int(round(cy + half))
    x_hi = int(round(cx + half))
    return y_lo, x_lo, y_hi, x_hi


class EyeDataset(Dataset):
    """
    Args:
        df: DataFrame with image/mask paths and metadata.
        data_root: Root directory for all relative paths.
        img_size: Resize images to (img_size, img_size).
        grayscale: If True (default), output 1-channel grayscale.
                   If False, output 3-channel RGB.
        augment: If True, apply the full RITnet-inspired augmentation pipeline.
        aug_version: "v1" | "v2" | "v3": which augmentation pipeline to use.
        online_crop: "none" | "tight" | "jitter": dataloader-time crop mode.
            - "none":   no crop (just resize). Use when the dataset is already cropped.
            - "tight":  crop around iris bbox so iris fills ~`crop_target_iris_pct` of the frame.
            - "jitter": tight + random centre offset + random zoom: simulates the noise of
                        a face/eye detector at deploy time. Only applied when augment=True.
        crop_target_iris_pct: Target fraction of crop occupied by the iris+pupil bbox (long side).
        crop_jitter_pct: Max centre offset as fraction of crop side (jitter mode only).
        crop_zoom_jitter: Random multiplier on crop_side: side *= 1 + U(-zj, +zj). (jitter mode only).
        framing_policy: "off" | "auto". When "auto", consult `src/framing.py`'s
            `FRAMING_POLICY` map per-sample (by `row["dataset"]`) and apply
            random_iris_framing on train / tight_iris_framing on val+test for
            whitelisted datasets (currently only `ubiris_v2_with_pupils`).
            Runs BEFORE the global `online_crop` step, so datasets not in the
            whitelist still follow `online_crop`. "off" preserves legacy behavior.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        img_size: int = 160,
        grayscale: bool = True,
        augment: bool = False,
        aug_version: str = "v3",
        online_crop: str = "none",
        crop_target_iris_pct: float = 0.35,
        crop_jitter_pct: float = 0.15,
        crop_zoom_jitter: float = 0.25,
        framing_policy: str = "off",
    ):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.grayscale = grayscale
        self.augment = augment
        if aug_version != "v3":
            raise ValueError("Esta publicación conserva únicamente aumentaciones v3")
        self.aug_version = aug_version
        self._aug_fn = apply_augmentations_v3

        if online_crop not in ("none", "tight", "jitter"):
            raise ValueError(f"Unknown online_crop '{online_crop}'. Use 'none', 'tight', or 'jitter'.")
        self.online_crop = online_crop
        self.crop_target_iris_pct = crop_target_iris_pct
        self.crop_jitter_pct = crop_jitter_pct
        self.crop_zoom_jitter = crop_zoom_jitter
        if framing_policy not in ("off", "auto"):
            raise ValueError(f"Unknown framing_policy '{framing_policy}'. Use 'off' or 'auto'.")
        self.framing_policy = framing_policy

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # --- Load ---
        img_path = self.data_root / row["rel_image_path"]
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        mask_path = self.data_root / row["rel_mask_path"]
        mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask_bgr is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # --- Mask → class indices ---
        converter = get_converter(row["dataset_format"])
        mask = converter(mask_bgr).astype(np.int64)  # (H, W) int64

        # --- Per-dataset framing policy (random_iris_framing for the
        # whitelisted uncropped datasets: see src/framing.py). Runs BEFORE
        # online_crop so when a UBIRIS uncropped sample gets framed, the
        # subsequent global crop is a no-op for it.
        framed_here = False
        if self.framing_policy == "auto":
            ds_name = str(row.get("dataset", ""))
            if FRAMING_POLICY.get(ds_name, "none") != "none":
                rng_f = np.random.default_rng(None if self.augment else 0)
                img_bgr, mask = apply_framing(
                    img_bgr, mask, ds_name, self.augment, rng_f,
                    target_iris_pct=self.crop_target_iris_pct,
                    jitter_pct=self.crop_jitter_pct,
                    zoom_jitter=self.crop_zoom_jitter,
                )
                framed_here = True

        # --- Online crop (around iris bbox): global fallback ---
        if not framed_here and self.online_crop != "none":
            jitter = (self.online_crop == "jitter") and self.augment
            rng = np.random.default_rng(None if jitter else 0)
            box = _online_crop_box(
                mask, self.crop_target_iris_pct, self.crop_jitter_pct,
                self.crop_zoom_jitter, rng, enable_jitter=jitter,
            )
            if box is not None:
                y0, x0, y1, x1 = box
                img_bgr = img_bgr[y0:y1, x0:x1]
                mask    = mask[y0:y1, x0:x1]

        # --- Resize (nearest for mask, linear for image) ---
        img_bgr = cv2.resize(img_bgr,  (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask    = cv2.resize(mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).astype(np.int64)

        # --- Preprocess: CLAHE + gamma: applied to every split ---
        img = preprocess(img_bgr, grayscale=self.grayscale)  # uint8 (H,W,1) or (H,W,3)

        # --- Augment: training only ---
        if self.augment:
            img, mask = self._aug_fn(img, mask)

        # --- To tensor ---
        img_t  = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C, H, W) [0,1]
        mask_t = torch.from_numpy(mask).long()                            # (H, W)

        return img_t, mask_t
