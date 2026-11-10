"""
Segmentation Metrics
=====================
Computes per-class and mean metrics for 3-class eye segmentation:
    class 0 = background
    class 1 = iris
    class 2 = pupil

Functions are designed to work on batches of predictions/targets (torch tensors).

Metrics:
    - IoU (Jaccard) per class and mean
    - Dice (F1) per class and mean
    - Center Distance (px): Euclidean distance between predicted and GT centroids
      (only meaningful for iris and pupil classes)
"""

from typing import Dict
import numpy as np
import torch
from scipy import ndimage as ndi


CLASS_NAMES_BY_NUM_CLASSES = {
    3: {0: "bg", 1: "iris",   2: "pupil"},
    4: {0: "bg", 1: "sclera", 2: "iris",  3: "pupil"},
}
CLASS_NAMES = CLASS_NAMES_BY_NUM_CLASSES[3]  # back-compat default
EPS = 1e-6
PUPIL_DETECT_MIN_PX = 5  # min predicted pupil pixels to count as "pupil detected"


def _names_for(num_classes: int) -> Dict[int, str]:
    return CLASS_NAMES_BY_NUM_CLASSES.get(num_classes, {i: f"c{i}" for i in range(num_classes)})


def _pupil_class_for(num_classes: int) -> int:
    """Pupil index for the active class scheme (used by centroid/etc.)."""
    return 3 if num_classes == 4 else 2


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> Dict[str, float]:
    """
    Compute IoU for each class.

    Returns dict with `iou_bg`, `iou_iris`, `iou_pupil` (3-class) OR
    `iou_bg`, `iou_sclera`, `iou_iris`, `iou_pupil` (4-class), plus `iou_mean`
    (excluding bg from the mean: for thesis-comparable numbers).
    """
    names = _names_for(num_classes)
    results = {}
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        iou = inter / (union + EPS)
        results[f"iou_{names[c]}"] = iou
        if c > 0:  # exclude bg from mean
            ious.append(iou)
    results["iou_mean"] = float(np.mean(ious)) if ious else 0.0
    # Also report iou_mean_iris_pupil so we can compare apples-to-apples vs the
    # 3-class baseline (which has no sclera).
    if num_classes == 4:
        iris_idx = 2; pupil_idx = 3
        results["iou_mean_iris_pupil"] = float(
            np.mean([results[f"iou_{names[iris_idx]}"], results[f"iou_{names[pupil_idx]}"]])
        )
    return results


def dice_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> Dict[str, float]:
    """Dice per class. See iou_per_class for keys."""
    names = _names_for(num_classes)
    results = {}
    dices = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        dice = (2 * inter) / (p.sum().item() + t.sum().item() + EPS)
        results[f"dice_{names[c]}"] = dice
        if c > 0:
            dices.append(dice)
    results["dice_mean"] = float(np.mean(dices)) if dices else 0.0
    if num_classes == 4:
        iris_idx = 2; pupil_idx = 3
        results["dice_mean_iris_pupil"] = float(
            np.mean([results[f"dice_{names[iris_idx]}"], results[f"dice_{names[pupil_idx]}"]])
        )
    return results


def _centroid(mask_np: np.ndarray) -> np.ndarray:
    """Return (y, x) centroid of a binary mask, or None if empty."""
    ys, xs = np.where(mask_np)
    if len(ys) == 0:
        return None
    return np.array([ys.mean(), xs.mean()])


def center_distance_per_class(
    pred: torch.Tensor, target: torch.Tensor, classes: tuple | None = None,
    num_classes: int = 3,
) -> Dict[str, float]:
    """
    Compute mean centroid Euclidean distance (in pixels) per class over batch.

    For images where either prediction or GT mask is empty, the sample is skipped.
    """
    names = _names_for(num_classes)
    if classes is None:
        # default: iris + pupil (the two anatomy-relevant classes)
        classes = (1, 2) if num_classes == 3 else (2, 3)
    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)
    results = {}

    for c in classes:
        dists = []
        for i in range(pred_np.shape[0]):
            p_centroid = _centroid(pred_np[i] == c)
            t_centroid = _centroid(target_np[i] == c)
            if p_centroid is not None and t_centroid is not None:
                dists.append(np.linalg.norm(p_centroid - t_centroid))
        results[f"centdist_{names[c]}"] = float(np.mean(dists)) if dists else float("nan")

    return results


def pupil_hole_baseline(pred: torch.Tensor, target: torch.Tensor,
                        num_classes: int = 3) -> Dict[str, float]:
    """For each sample with GT pupil, predict pupil as the hole inside the
    predicted iris (fill_holes(pred_iris) - pred_iris) and compute IoU vs GT.
    """
    iris_idx = 1 if num_classes == 3 else 2
    pupil_idx = 2 if num_classes == 3 else 3
    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)
    ious = []
    for i in range(pred_np.shape[0]):
        gt_pupil = target_np[i] == pupil_idx
        if gt_pupil.sum() == 0:
            continue
        pred_iris = pred_np[i] == iris_idx
        inferred = np.logical_and(ndi.binary_fill_holes(pred_iris), ~pred_iris)
        inter = np.logical_and(inferred, gt_pupil).sum()
        union = np.logical_or(inferred, gt_pupil).sum()
        ious.append(inter / (union + EPS))
    return {"hole_pupil_iou": float(np.mean(ious)) if ious else float("nan")}


def pupil_detection_stats(pred: torch.Tensor, target: torch.Tensor,
                          min_px: int = PUPIL_DETECT_MIN_PX,
                          num_classes: int = 3) -> Dict[str, float]:
    """Per-batch pupil presence stats:

    - pupil_detect_rate: % of samples with GT pupil where pred has ≥ min_px pupil pixels.
                          Measures "did we find the pupil at all?"
    - pupil_fp_rate:     % of samples without GT pupil where pred predicts pupil.
                          False-positive rate. should be near 0 on iris-only datasets.

    NaN when the corresponding sub-population is empty in the batch.
    """
    pupil_idx = 2 if num_classes == 3 else 3
    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)
    n_with = n_without = 0
    detected = false_pos = 0
    for i in range(pred_np.shape[0]):
        gt = (target_np[i] == pupil_idx).sum()
        pr = (pred_np[i] == pupil_idx).sum()
        if gt > 0:
            n_with += 1
            if pr >= min_px:
                detected += 1
        else:
            n_without += 1
            if pr >= min_px:
                false_pos += 1
    return {
        "pupil_detect_rate": detected / n_with if n_with else float("nan"),
        "pupil_fp_rate": false_pos / n_without if n_without else float("nan"),
        "n_with_gt_pupil": float(n_with),
        "n_without_gt_pupil": float(n_without),
    }


def iou_iris_split_by_gt_pupil(pred: torch.Tensor, target: torch.Tensor,
                                num_classes: int = 3) -> Dict[str, float]:
    """IoU for the iris class, split by whether GT has a pupil annotation."""
    iris_idx = 1 if num_classes == 3 else 2
    pupil_idx = 2 if num_classes == 3 else 3
    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)
    ious_with, ious_without = [], []
    for i in range(pred_np.shape[0]):
        gt_iris = target_np[i] == iris_idx
        pr_iris = pred_np[i] == iris_idx
        if gt_iris.sum() == 0 and pr_iris.sum() == 0:
            iou = 1.0
        else:
            inter = np.logical_and(gt_iris, pr_iris).sum()
            union = np.logical_or(gt_iris, pr_iris).sum()
            iou = inter / (union + EPS)
        if (target_np[i] == pupil_idx).sum() > 0:
            ious_with.append(iou)
        else:
            ious_without.append(iou)
    return {
        "iou_iris_with_gt_pupil":    float(np.mean(ious_with))    if ious_with    else float("nan"),
        "iou_iris_no_gt_pupil":      float(np.mean(ious_without)) if ious_without else float("nan"),
    }


def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                    num_classes: int = 3) -> Dict[str, float]:
    """Compute all metrics for a batch. Pass num_classes=4 for sclera-aware."""
    metrics = {}
    metrics.update(iou_per_class(pred, target, num_classes=num_classes))
    metrics.update(dice_per_class(pred, target, num_classes=num_classes))
    metrics.update(center_distance_per_class(pred, target, num_classes=num_classes))
    metrics.update(pupil_hole_baseline(pred, target, num_classes=num_classes))
    metrics.update(pupil_detection_stats(pred, target, num_classes=num_classes))
    metrics.update(iou_iris_split_by_gt_pupil(pred, target, num_classes=num_classes))
    return metrics


def aggregate_metrics(list_of_metrics: list) -> Dict[str, float]:
    """
    Combine a list of per-batch metric dicts into a single dict.
    - Keys starting with 'n_' are summed across batches (counts).
    - All other keys are averaged. NaN values are ignored in the mean.
    """
    if not list_of_metrics:
        return {}
    keys = list_of_metrics[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in list_of_metrics if not np.isnan(m.get(k, float("nan")))]
        if not vals:
            result[k] = float("nan")
        elif k.startswith("n_"):
            result[k] = float(np.sum(vals))
        else:
            result[k] = float(np.mean(vals))
    return result
