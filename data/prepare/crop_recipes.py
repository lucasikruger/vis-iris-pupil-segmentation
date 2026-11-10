import cv2
import numpy as np

def zoom_polyu(img: np.ndarray, mask: np.ndarray,
              zoom_factor: float, target_w: int, target_h: int
              ) -> tuple[np.ndarray, np.ndarray] | None:
    """Crop (img, mask) so iris ∪ pupil bbox fills ~zoom_factor of the crop's
    longest dimension, with the (target_w, target_h) aspect ratio."""
    H, W = mask.shape[:2]
    eye = (mask[..., 1] > 0) | (mask[..., 0] > 0)  # G=iris, B=pupil
    if not eye.any():
        return None
    ys, xs = np.where(eye)
    bx0, by0 = int(xs.min()), int(ys.min())
    bx1, by1 = int(xs.max()), int(ys.max())
    bw, bh = bx1 - bx0, by1 - by0
    bcx, bcy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    target_ar = target_w / target_h
    longest = max(bw, bh)
    if longest <= 0:
        return None
    if bw >= bh:
        crop_w = longest / zoom_factor
        crop_h = crop_w / target_ar
    else:
        crop_h = longest / zoom_factor
        crop_w = crop_h * target_ar
    if crop_w < bw:
        crop_w = float(bw); crop_h = crop_w / target_ar
    if crop_h < bh:
        crop_h = float(bh); crop_w = crop_h * target_ar
    x0 = bcx - crop_w / 2.0
    y0 = bcy - crop_h / 2.0
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > W: x0 -= (x1 - W); x1 = W
    if y1 > H: y0 -= (y1 - H); y1 = H
    x0i, y0i = max(0, int(round(x0))), max(0, int(round(y0)))
    x1i, y1i = min(W, int(round(x1))), min(H, int(round(y1)))
    if x1i - x0i < 4 or y1i - y0i < 4:
        return None
    crop_img  = img[y0i:y1i,  x0i:x1i]
    crop_mask = mask[y0i:y1i, x0i:x1i]
    out_img  = cv2.resize(crop_img,  (target_w, target_h), interpolation=cv2.INTER_AREA)
    out_mask = cv2.resize(crop_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return out_img, out_mask

def zoom_ubiris(img: np.ndarray, mask: np.ndarray,
              zoom_factor: float, target_w: int, target_h: int
              ) -> tuple[np.ndarray, np.ndarray] | None:
    """Crop image+mask so the iris bbox occupies ~zoom_factor of the crop's
    longest dimension. Crop has the same aspect ratio as (target_w, target_h)
    so the final resize doesn't distort. Returns (img_out, mask_out) at the
    requested target size, or None if no iris is present."""
    H, W = mask.shape[:2]
    # G channel = iris ∪ pupil (per merge_pupils invariant). B = pupil
    eye = (mask[..., 1] > 0) | (mask[..., 0] > 0)
    if not eye.any():
        return None
    ys, xs = np.where(eye)
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bw, bh = bx1 - bx0, by1 - by0
    bcx, bcy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    target_ar = target_w / target_h

    # Required crop dimensions so that the longest of (bw, bh) fills zoom_factor.
    longest = max(bw, bh)
    if longest <= 0:
        return None
    if bw >= bh:
        crop_w = longest / zoom_factor
        crop_h = crop_w / target_ar
    else:
        crop_h = longest / zoom_factor
        crop_w = crop_h * target_ar
    # Make sure both bbox dimensions fit (i.e. neither side gets cut off when
    # the iris bbox is unusually elongated against the target aspect ratio).
    if crop_w < bw:
        crop_w = float(bw); crop_h = crop_w / target_ar
    if crop_h < bh:
        crop_h = float(bh); crop_w = crop_h * target_ar

    x0 = bcx - crop_w / 2.0
    y0 = bcy - crop_h / 2.0
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    # Shift the window inside the image if needed (preserves dims).
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > W: x0 -= (x1 - W); x1 = W
    if y1 > H: y0 -= (y1 - H); y1 = H
    x0i, y0i = max(0, int(round(x0))), max(0, int(round(y0)))
    x1i, y1i = min(W, int(round(x1))), min(H, int(round(y1)))
    if x1i - x0i < 4 or y1i - y0i < 4:
        return None

    crop_img  = img[y0i:y1i,  x0i:x1i]
    crop_mask = mask[y0i:y1i, x0i:x1i]
    out_img  = cv2.resize(crop_img,  (target_w, target_h), interpolation=cv2.INTER_AREA)
    out_mask = cv2.resize(crop_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return out_img, out_mask

def tight_eyelid_crop(img, mobius_bgr, eyelid_poly_native, pad_frac=0.10):
    """Crop image+mask to a square bbox around the eyelid polygon + pad.
    If no polygon, return inputs unchanged.

    eyelid_poly_native: polygon coords in the SOURCE image's native pixel space.
    """
    H, W = img.shape[:2]
    if eyelid_poly_native is None or len(eyelid_poly_native) < 3:
        return img, mobius_bgr, None
    poly = np.array(eyelid_poly_native, dtype=np.float32)
    x0 = max(0, int(poly[:, 0].min()))
    y0 = max(0, int(poly[:, 1].min()))
    x1 = min(W, int(np.ceil(poly[:, 0].max())))
    y1 = min(H, int(np.ceil(poly[:, 1].max())))
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return img, mobius_bgr, None
    # Pad
    px = int(round(pad_frac * bw)); py = int(round(pad_frac * bh))
    x0 = max(0, x0 - px); x1 = min(W, x1 + px)
    y0 = max(0, y0 - py); y1 = min(H, y1 + py)
    # Square it (use the larger side, centered)
    cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0)
    sx0 = max(0, int(round(cx - side / 2)))
    sy0 = max(0, int(round(cy - side / 2)))
    sx1 = min(W, sx0 + side); sy1 = min(H, sy0 + side)
    img_c = img[sy0:sy1, sx0:sx1].copy()
    msk_c = mobius_bgr[sy0:sy1, sx0:sx1].copy()
    # Re-localize polygon
    poly_in_crop = poly.copy()
    poly_in_crop[:, 0] -= sx0
    poly_in_crop[:, 1] -= sy0
    return img_c, msk_c, poly_in_crop

def resize_to(img, mobius_bgr, poly, out_size):
    """Resize image (LINEAR) + mask (NEAREST), scale polygon coords."""
    H, W = img.shape[:2]
    img_r = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    msk_r = cv2.resize(mobius_bgr, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    poly_r = None
    if poly is not None:
        sx = out_size / W; sy = out_size / H
        poly_r = poly.copy()
        poly_r[:, 0] *= sx; poly_r[:, 1] *= sy
    return img_r, msk_r, poly_r
