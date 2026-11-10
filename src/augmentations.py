"""
Eye-Specific Augmentation Pipeline
====================================
Adapted from the RITnet augmentation strategy (Chaudhary et al., 2019) and
extended for visible-light RGB webcam images.

RITnet augmentations kept / adapted:
  - Random horizontal flip (50%)
  - Random translation (40%, 2-38 px)
  - Gaussian blur (20%, kernel 5-9, sigma 2-6)
  - Random lines: eyelash / hair occlusion simulation (20%, 1-9 lines)
  - Starburst: corneal specular reflection simulation (20%)

Added for visible-light (NIR → RGB gap mitigation):
  - Brightness/contrast jitter (35%)
  - Gaussian noise (20%)

Preprocessing applied to ALL splits (train + val + test):
  - CLAHE: adaptive contrast enhancement
  - Gamma correction (factor 0.8): matches RITnet baseline

All functions operate on uint8 numpy arrays.
Augmentation functions that move pixels (translation) apply the same transform
to image and mask to keep them aligned. Appearance-only augmentations
(blur, lines, starburst, brightness, noise) touch only the image.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Preprocessing: applied to every split
# ---------------------------------------------------------------------------

def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction via a lookup table (fast, uint8 safe)."""
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def preprocess(img_bgr: np.ndarray, grayscale: bool) -> np.ndarray:
    """
    Convert a BGR uint8 image to grayscale (H×W×1) or RGB (H×W×3)
    with CLAHE contrast enhancement and gamma correction applied.

    This is applied to ALL splits: it is not an augmentation.

    Args:
        img_bgr: OpenCV-loaded image, shape (H, W, 3), BGR, uint8.
        grayscale: If True, returns (H, W, 1). else (H, W, 3) RGB.

    Returns:
        Preprocessed image as uint8 numpy array.
    """
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    if grayscale:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # (H, W)
        gray = clahe.apply(gray)
        gray = _apply_gamma(gray, gamma=0.8)
        return gray[:, :, np.newaxis]                       # (H, W, 1)
    else:
        # CLAHE on the L channel of LAB to preserve colour balance
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        rgb = _apply_gamma(rgb, gamma=0.8)
        return rgb                                          # (H, W, 3)


# ---------------------------------------------------------------------------
# Geometric augmentations: applied to image AND mask
# ---------------------------------------------------------------------------

def aug_flip(img: np.ndarray, mask: np.ndarray) -> tuple:
    """Random horizontal flip (p=0.5). Safe for (H,W,1) and (H,W,3)."""
    if np.random.rand() < 0.5:
        img  = img[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
    return img, mask




# ---------------------------------------------------------------------------
# Appearance augmentations: applied to image only
# ---------------------------------------------------------------------------

def _white_color(img: np.ndarray):
    """Return white colour scalar matching image channels."""
    return 255 if img.ndim == 2 or img.shape[2] == 1 else (255, 255, 255)


def _squeeze(img: np.ndarray):
    """Squeeze (H,W,1) → (H,W) for OpenCV ops that drop channel dim."""
    squeezed = img.ndim == 3 and img.shape[2] == 1
    return img[:, :, 0] if squeezed else img, squeezed


def _unsqueeze(img: np.ndarray, was_squeezed: bool) -> np.ndarray:
    """Restore (H,W) → (H,W,1) if it was squeezed."""
    if was_squeezed and img.ndim == 2:
        return img[:, :, np.newaxis]
    return img


def aug_gaussian_blur(img: np.ndarray) -> np.ndarray:
    """Gaussian blur (kernel 5-9, sigma 2-6): simulates motion blur / defocus."""
    k = int(np.random.choice([5, 7, 9]))
    sigma = np.random.uniform(2.0, 6.0)
    img2d, sq = _squeeze(img)
    out = cv2.GaussianBlur(img2d, (k, k), sigma)
    return _unsqueeze(out, sq)


def aug_random_lines(img: np.ndarray, n_max: int = 9) -> np.ndarray:
    """
    Simulate eyelash and hair occlusion with random white line segments.
    Lines originate near the image centre and radiate outward.
    """
    img2d, sq = _squeeze(img)
    img2d = img2d.copy()
    h, w = img2d.shape[:2]
    color = 255 if img2d.ndim == 2 else (255, 255, 255)
    n = np.random.randint(1, n_max + 1)
    for _ in range(n):
        cx = int(np.random.uniform(0.3, 0.7) * w)
        cy = int(np.random.uniform(0.3, 0.7) * h)
        angle = np.random.uniform(0, np.pi)
        length = np.random.randint(50, min(200, max(h, w)))
        dx = int(np.cos(angle) * length)
        dy = int(np.sin(angle) * length)
        cv2.line(img2d, (cx - dx, cy - dy), (cx + dx, cy + dy), color, 4)
    return _unsqueeze(img2d, sq)


def aug_starburst(img: np.ndarray, n_spokes: int = 12) -> np.ndarray:
    """
    Simulate corneal specular reflection (starburst pattern).
    Draws a bright central spot with radial spokes at a random position
    in the central region of the eye crop.

    Particularly relevant for visible-light images where specular
    highlights are common (unlike NIR).
    """
    img2d, sq = _squeeze(img)
    img2d = img2d.copy()
    h, w = img2d.shape[:2]
    color = 255 if img2d.ndim == 2 else (255, 255, 255)

    cx = int(np.random.uniform(0.25, 0.75) * w)
    cy = int(np.random.uniform(0.25, 0.75) * h)

    # Central bright spot
    spot_r = np.random.randint(2, 6)
    cv2.circle(img2d, (cx, cy), spot_r, color, -1)

    # Radial spokes
    max_len = min(h, w) // 3
    for i in range(n_spokes):
        angle = (2 * np.pi * i / n_spokes) + np.random.uniform(-0.15, 0.15)
        length = np.random.randint(8, max(9, max_len))
        ex = int(cx + np.cos(angle) * length)
        ey = int(cy + np.sin(angle) * length)
        thickness = np.random.randint(1, 3)
        cv2.line(img2d, (cx, cy), (ex, ey), color, thickness)

    return _unsqueeze(img2d, sq)


def aug_brightness_contrast(img: np.ndarray,
                             brightness_range: tuple = (-40, 40),
                             contrast_range: tuple = (0.75, 1.25)) -> np.ndarray:
    """
    Random brightness shift and contrast scaling.
    Particularly important for visible-light where scene lighting varies widely.
    """
    alpha = np.random.uniform(*contrast_range)
    beta  = int(np.random.randint(brightness_range[0], brightness_range[1] + 1))
    img2d, sq = _squeeze(img)
    out = cv2.convertScaleAbs(img2d, alpha=alpha, beta=beta)
    return _unsqueeze(out, sq)


def aug_gaussian_noise(img: np.ndarray, std_range: tuple = (5, 25)) -> np.ndarray:
    """Add zero-mean Gaussian noise: simulates sensor noise."""
    std = np.random.uniform(*std_range)
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main augmentation entry point
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# v2: extensions for the second experiment
# ---------------------------------------------------------------------------
# Added on top of v1 baseline:
#   1. Eyelid occlusion      : horizontal half-plane erase (simulates blink / lid drop)
#   2. Vertical occlusion    : vertical half-plane erase (simulates obscure side)
#   3. Pupil resize          : physiological pupil dilation/constriction
#   4. Aggressive translation: larger displacements + 8 directions (cardinal + diagonal)
#   5. Skin tone shift       : darker/lighter skin tone simulation (RGB only)
#   6. Iris darken           : simulate dark iris colors (brown)
#   7. Bounded rotation      : small in-plane rotations (±15°)
# References inline in each function docstring.




def aug_vertical_occlusion(img: np.ndarray, mask: np.ndarray) -> tuple:
    """Oclusión lateral REDONDEADA (como un párpado pero de costado): nariz, marco,
    dedo, etc. Entra desde la izquierda o la derecha con un borde parabólico (más
    profundo en el centro vertical, retrocede hacia arriba/abajo) y NUNCA pasa la
    mitad de la imagen. Curvatura/penetración random, pero SIEMPRE ocluye algo del
    iris (ajusta la penetración). Devuelve (img, mask)."""
    h, w = img.shape[:2]
    fg = _mask_fg(mask)
    bb = _bbox_of(fg)
    iris_pupil = (mask == 1) | (mask == 2)
    total = int(iris_pupil.sum())
    if bb is None or total < 10:
        return img, mask
    _, x0_bb, _, x1_bb = bb
    bb_w = max(1, x1_bb - x0_bb)

    left = np.random.rand() < 0.5
    norm = ((np.arange(h) - h / 2.0) / (h / 2.0)) ** 2     # 0 en el centro vertical → 1 arriba/abajo
    amp = float(np.random.uniform(0.4, 1.2)) * bb_w         # redondeo
    pen = float(np.random.uniform(0.20, 0.60)) * bb_w
    xx = np.indices((h, w))[1]
    erase = None
    x_cut = None
    for _ in range(8):
        # como el párpado pero de costado: MÁS profundo en los bordes (arriba/abajo),
        # abierto (retrocede) en el centro vertical. No pasa la mitad.
        if left:
            x_edge = min(0.48 * w, x0_bb + pen)             # punto más profundo (en arriba/abajo)
            x_cut = (x_edge - amp) + amp * norm             # centro retrocede a x_edge-amp
        else:
            x_edge = max(0.52 * w, x1_bb - pen)
            x_cut = (x_edge + amp) - amp * norm
        x_cut = np.clip(np.round(x_cut).astype(np.int32), 0, w - 1)
        erase = (xx < x_cut[:, None]) if left else (xx > x_cut[:, None])
        frac = float((erase & iris_pupil).sum()) / total
        if 0.03 <= frac <= 0.55:
            break
        pen += (0.15 * bb_w if frac < 0.03 else -0.15 * bb_w)
        pen = max(0.0, pen)

    edge_xs = np.clip(x_cut + (1 if left else -1), 0, w - 1)
    edge_ys = np.arange(h)
    img_out = img.copy()
    mask_out = mask.copy()
    if img.ndim == 2:
        img_out[erase] = int(np.median(img[edge_ys, edge_xs]))
    elif img.shape[2] == 1:
        img_out[..., 0][erase] = int(np.median(img[edge_ys, edge_xs, 0]))
    else:
        img_out[erase] = np.median(img[edge_ys, edge_xs, :], axis=0).astype(img.dtype)
    mask_out[erase] = 0
    return img_out, mask_out


def aug_pupil_resize(img: np.ndarray, mask: np.ndarray,
                     scale_range: tuple = (0.6, 1.5)) -> tuple:
    """
    Scale the pupil by a random factor: simulates physiological pupil
    dilation/constriction (pupil ratio varies ~0.15-0.45 with lighting).

    Strategy:
      1. Find pupil mask (class 2), get centroid + bbox.
      2. Reset the original pupil region → iris (class 1) in the mask, and
         repaint that area with the median iris colour in the image (so the
         "old" pupil doesn't leave a hole).
      3. Resize the pupil binary patch by `scale`, paste centred at the
         original centroid → new pupil region.
      4. Where the new pupil mask is set, set image pixels to the median
         pupil colour (very dark).

    Side effect: image-pupil-edges may show slight aliasing because the
    fill colour is uniform. Acceptable for a segmentation augmentation.
    """
    pupil_mask = (mask == 2)
    if pupil_mask.sum() < 10:
        return img, mask  # no pupil → no-op

    h, w = img.shape[:2]
    iris_mask = (mask == 1)

    # Original pupil + iris colours (for repaint)
    if img.ndim == 3 and img.shape[2] == 3:
        old_pupil_color = np.median(img[pupil_mask], axis=0).astype(img.dtype)
        iris_color = (np.median(img[iris_mask], axis=0).astype(img.dtype)
                      if iris_mask.sum() > 10 else old_pupil_color)
    else:
        flat = img[..., 0] if img.ndim == 3 else img
        old_pupil_color = int(np.median(flat[pupil_mask]))
        iris_color = (int(np.median(flat[iris_mask])) if iris_mask.sum() > 10 else old_pupil_color)

    ys, xs = np.where(pupil_mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    ph, pw = y1 - y0, x1 - x0

    scale = float(np.random.uniform(*scale_range))
    new_h = max(2, int(round(ph * scale)))
    new_w = max(2, int(round(pw * scale)))

    pup_patch = pupil_mask[y0:y1, x0:x1].astype(np.uint8)
    new_pup = cv2.resize(pup_patch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Step 1: clear original pupil region → iris (in mask + image)
    mask_out = mask.copy()
    img_out = img.copy()
    mask_out[pupil_mask] = 1  # to iris
    if img.ndim == 2:
        img_out[pupil_mask] = iris_color
    elif img.shape[2] == 1:
        img_out[..., 0][pupil_mask] = iris_color
    else:
        img_out[pupil_mask] = iris_color

    # Step 2: paste new pupil centred at (cy, cx)
    y_top  = cy - new_h // 2
    x_left = cx - new_w // 2
    # Clamp into image, slice patch accordingly
    yt_c = max(0, y_top);   xl_c = max(0, x_left)
    yb_c = min(h, y_top + new_h); xr_c = min(w, x_left + new_w)
    dy = yt_c - y_top; dx = xl_c - x_left
    patch_h = yb_c - yt_c; patch_w = xr_c - xl_c
    if patch_h <= 0 or patch_w <= 0:
        return img, mask
    patch = new_pup[dy:dy + patch_h, dx:dx + patch_w]
    where = patch > 0

    region_mask = mask_out[yt_c:yb_c, xl_c:xr_c]
    region_mask[where] = 2  # pupil

    if img.ndim == 2:
        img_out[yt_c:yb_c, xl_c:xr_c][where] = old_pupil_color
    elif img.shape[2] == 1:
        img_out[yt_c:yb_c, xl_c:xr_c, 0][where] = old_pupil_color
    else:
        img_out[yt_c:yb_c, xl_c:xr_c][where] = old_pupil_color

    return img_out, mask_out


def _border_fill_color(img: np.ndarray) -> tuple:
    """Median colour of the outermost 2-pixel border ring: sensible fill for
    extrapolated regions (translation / rotation / zoom-out). Avoids the
    mirror-eye effect of BORDER_REFLECT and the unnatural black of CONSTANT 0.
    """
    if img.ndim == 2:
        ring = np.concatenate([img[:2].ravel(), img[-2:].ravel(),
                                img[:, :2].ravel(), img[:, -2:].ravel()])
        return int(np.median(ring))
    if img.shape[2] == 1:
        ring = np.concatenate([img[:2, :, 0].ravel(), img[-2:, :, 0].ravel(),
                                img[:, :2, 0].ravel(), img[:, -2:, 0].ravel()])
        return int(np.median(ring))
    # 3-channel: per-channel median
    ring = np.concatenate([img[:2].reshape(-1, 3),  img[-2:].reshape(-1, 3),
                            img[:, :2].reshape(-1, 3), img[:, -2:].reshape(-1, 3)])
    return tuple(int(c) for c in np.median(ring, axis=0))


def aug_translation_v2(img: np.ndarray, mask: np.ndarray,
                       max_shift_frac: float = 0.45) -> tuple:
    """
    Aggressive translation: randomises iris position in the frame.

      - Up to ±max_shift_frac × min(H, W). On 160×160 with default 0.45 → 72 px.
      - 8 directions (cardinal + diagonal). diagonals scaled 0.7× per axis.
      - Padding: **solid border colour** (median of image border ring) for the
        image, mask padded with background (class 0). NO REFLECT: reflect was
        creating mirror-eye artifacts that the model had no reason to learn.

    Combined with aug_zoom this teaches position+scale invariance.
    """
    img2d, sq = _squeeze(img)
    h, w = img2d.shape[:2]
    max_px = max(2, int(min(h, w) * max_shift_frac))
    shift = int(np.random.randint(2, max_px + 1))
    direction = np.random.randint(8)
    dirs = [(0, -1), (0, 1), (-1, 0), (1, 0),
            (-1, -1), (1, -1), (-1, 1), (1, 1)]
    dx_sign, dy_sign = dirs[direction]
    if dx_sign != 0 and dy_sign != 0:
        s = int(shift * 0.7)
        tx, ty = dx_sign * s, dy_sign * s
    else:
        tx, ty = dx_sign * shift, dy_sign * shift
    M = np.float32([[1, 0, tx], [0, 1, ty]])

    fill = _border_fill_color(img)
    out2d = cv2.warpAffine(img2d, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    mask_out = cv2.warpAffine(mask.astype(np.uint8), M, (w, h),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0).astype(np.int64)
    return _unsqueeze(out2d, sq), mask_out


def aug_rotation(img: np.ndarray, mask: np.ndarray,
                 max_angle: float = 25.0) -> tuple:
    """
    Random in-plane rotation bounded to ±max_angle degrees. Padding = solid
    border-median colour (image) and class 0 (mask). NO REFLECT.
    """
    img2d, sq = _squeeze(img)
    h, w = img2d.shape[:2]
    angle = float(np.random.uniform(-max_angle, max_angle))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)

    fill = _border_fill_color(img)
    out2d = cv2.warpAffine(img2d, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    mask_out = cv2.warpAffine(mask.astype(np.uint8), M, (w, h),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0).astype(np.int64)
    return _unsqueeze(out2d, sq), mask_out


def aug_zoom(img: np.ndarray, mask: np.ndarray,
             scale_range: tuple = (0.55, 1.30)) -> tuple:
    """
    Random zoom in or out. Zoom-out padding = solid border-median colour.
    """
    h, w = img.shape[:2]
    scale = float(np.random.uniform(*scale_range))
    new_h, new_w = max(4, int(round(h * scale))), max(4, int(round(w * scale)))
    img2d, sq = _squeeze(img)

    img_r = cv2.resize(img2d, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask.astype(np.uint8), (new_w, new_h),
                        interpolation=cv2.INTER_NEAREST).astype(np.int64)

    if scale >= 1.0:
        # Center crop back to (h, w): no padding needed.
        y_off = (new_h - h) // 2
        x_off = (new_w - w) // 2
        img_out = img_r[y_off:y_off + h, x_off:x_off + w]
        mask_out = mask_r[y_off:y_off + h, x_off:x_off + w]
    else:
        # Pad with solid border colour (NOT reflect: was producing mirror eyes).
        py_t = (h - new_h) // 2
        py_b = h - new_h - py_t
        px_l = (w - new_w) // 2
        px_r = w - new_w - px_l
        fill = _border_fill_color(img)
        img_out = cv2.copyMakeBorder(img_r, py_t, py_b, px_l, px_r,
                                      cv2.BORDER_CONSTANT, value=fill)
        # Mask: pad with class 0 (background)
        mask_out = np.pad(mask_r, ((py_t, py_b), (px_l, px_r)),
                           mode="constant", constant_values=0)

    if img_out.shape[:2] != (h, w):
        img_out = cv2.resize(img_out, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_out = cv2.resize(mask_out.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(np.int64)
    return _unsqueeze(img_out, sq), mask_out


def aug_skin_tone(img: np.ndarray, factor_range: tuple = (-0.5, 0.5)) -> np.ndarray:
    """
    Shift the image toward darker or lighter skin tone (RGB only: no-op on
    grayscale). Implements a per-channel multiplicative scaling inspired by
    Fitzpatrick scale variation: darker skin reduces all channels, with blue
    reduced slightly more (warmer brown). lighter skin increases all channels
    with blue increasing slightly more (pinker).

    factor: -0.5 = much darker, 0 = no change, +0.5 = much lighter.

    Reference: Buolamwini & Gebru 2018: "Gender Shades": highlights skin
    tone bias in face/eye datasets. explicit augmentation across the spectrum
    is a known mitigation.
    """
    if img.ndim == 2 or img.shape[2] == 1:
        return img  # no-op for grayscale
    factor = float(np.random.uniform(*factor_range))
    img_f = img.astype(np.float32)
    if factor < 0:
        # Darker skin: warmer (less blue, less green)
        scales = np.array([1 + factor * 0.55,   # R
                            1 + factor * 0.70,   # G
                            1 + factor * 0.90])  # B
    else:
        # Lighter skin: pinker (slightly more blue)
        scales = np.array([1 + factor * 0.40,   # R
                            1 + factor * 0.40,   # G
                            1 + factor * 0.50])  # B
    img_f = img_f * scales[None, None, :]
    return np.clip(img_f, 0, 255).astype(np.uint8)


def aug_iris_darken(img: np.ndarray, mask: np.ndarray,
                    factor_range: tuple = (0.35, 0.85)) -> np.ndarray:
    """
    Darken the iris AND pupil pixels by a random factor.

    Simulates the variation of natural iris colours: blue/green/hazel irises
    are bright, brown/black irises are much darker. Most public iris datasets
    are skewed toward lighter eyes (Caucasian subjects), so darkening as
    augmentation helps cover the dark-iris distribution. The pupil is darkened
    together with the iris so the ojo interno queda coherente (evita el borde
    raro iris-oscuro / pupila-clara).

    Reference: Iris colour distribution by Frost (2006): most studies report
    ~80% brown/dark globally vs lighter-skewed in Western datasets.
    """
    region_mask = (mask == 1) | (mask == 2)     # iris + pupila
    if region_mask.sum() < 10:
        return img
    factor = float(np.random.uniform(*factor_range))
    img_out = img.copy()
    if img_out.ndim == 2:
        img_out[region_mask] = (img_out[region_mask].astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
    elif img_out.shape[2] == 1:
        img_out[..., 0][region_mask] = (img_out[..., 0][region_mask].astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
    else:
        for c in range(img_out.shape[2]):
            img_out[..., c][region_mask] = (img_out[..., c][region_mask].astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
    return img_out


_IRIS_PALETTE = [          # RGB: colores de iris posibles
    (95, 60, 30),          # marrón
    (48, 34, 24),          # negro / marrón muy oscuro
    (85, 128, 72),         # verde
    (78, 120, 185),        # azul
    (135, 138, 140),       # gris
    (178, 128, 55),        # ámbar / miel
    (128, 98, 55),         # avellana
    (168, 55, 55),         # rojo (no natural, pedido)
]


def aug_iris_color(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cambia el color del iris a uno posible (marrón, negro, verde, azul, gris,
    ámbar, avellana o rojo). Conserva la textura usando la luminancia normalizada
    del iris, deja la pupila oscura y preserva los reflejos brillantes. Solo imagen
    (usa la máscara). En gris no hace nada."""
    iris = (mask == 1)
    if iris.sum() < 30 or img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return img
    target = np.array(_IRIS_PALETTE[np.random.randint(len(_IRIS_PALETTE))], np.float32)
    rgb = img[..., :3].astype(np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    li = lum[iris]
    lo, hi = np.percentile(li, 5), np.percentile(li, 95)
    ln = np.clip((lum - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    shade = (0.30 + 0.85 * ln)[..., None]              # piso para que no quede negro
    new = np.clip(shade * target[None, None, :], 0, 255)
    out = img.copy().astype(np.float32)
    a = 0.85
    out[iris] = new[iris] * a + rgb[iris] * (1 - a)
    spec = iris & (lum > 208)                           # conservar reflejos (catchlights)
    out[spec] = rgb[spec]
    return np.clip(out, 0, 255).astype(img.dtype)


def aug_erase_iris(img: np.ndarray, mask: np.ndarray) -> tuple:
    """Borra el iris+pupila (poco común): como si NO estuvieran.

    Pinta la región del iris+pupila de blanco: usa el color casi-blanco de la
    esclera a los lados del iris, empujado hacia el blanco puro, con ruido suave.
    Inventa así el fondo donde estaba el iris. La MÁSCARA se vacía en esa región
    (queda como background). Devuelve (img, mask). Para hard-negatives / robustez.
    """
    fg = (mask == 1) | (mask == 2)
    if fg.sum() < 30:
        return img, mask
    h, w = img.shape[:2]
    region = cv2.dilate((fg.astype(np.uint8) * 255), np.ones((5, 5), np.uint8), iterations=1) > 0

    ys, xs = np.where(fg)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    yc = (y0 + y1) // 2
    r = max(2, (x1 - x0) // 8)
    samp = []
    for cx in (x0 - r, x1 + r):                     # esclera a izquierda y derecha del iris
        cx = int(np.clip(cx, 0, w - 1))
        ylo, yhi = max(0, yc - r), min(h, yc + r + 1)
        xlo, xhi = max(0, cx - r), min(w, cx + r + 1)
        p = img[ylo:yhi, xlo:xhi]
        if p.size:
            samp.append(p.reshape(-1, p.shape[-1]) if p.ndim == 3 else p.reshape(-1, 1))
    c = 1 if img.ndim == 2 else img.shape[2]
    if samp:
        px = np.concatenate(samp, axis=0).astype(np.float32)
        br = px.mean(axis=1)
        white = px[br >= np.percentile(br, 60)].mean(axis=0)
        k = float(np.random.uniform(0.3, 0.7))
        white = white * (1.0 - k) + 255.0 * k
    else:
        white = np.full(c, 235.0, np.float32)

    out = img.astype(np.float32).copy()
    if img.ndim == 2:
        base = np.full((h, w), float(white[0]), np.float32) + np.random.normal(0, 5, (h, w)).astype(np.float32)
        out[region] = np.clip(base, 0, 255)[region]
    else:
        base = np.ones((h, w, c), np.float32) * white[:c][None, None, :] + np.random.normal(0, 5, (h, w, c)).astype(np.float32)
        out[region] = np.clip(base, 0, 255)[region]

    mask_out = mask.copy()
    mask_out[fg] = 0                                # el iris ya no está → bg
    return out.astype(img.dtype), mask_out


# ---------------------------------------------------------------------------
# Oclusión con assets reales transparentes (src/assets/occlusion/*)
# ---------------------------------------------------------------------------

# reglas por categoría (mismas que la página /occlusion). occl=True → tapa la máscara.
_OCC_RULES = {
    "pestañas": dict(scMin=.50, scMax=.80, rotMax=10, dx=.12, dy=.06, bx=.5, by=.34,
                     mirror=True, flipY=True, cover=False, blend="normal", occl=False, opac=(.50, .80)),
    "cejas":    dict(scMin=.45, scMax=.68, rotMax=12, dx=.12, dy=.06, bx=.5, by=.26,
                     mirror=True, flipY=False, cover=False, blend="normal", occl=False, opac=(.55, .85)),
    "anteojos": dict(scMin=2.2, scMax=3.0, rotMax=8, dx=.46, dy=.10, bx=.5, by=.5,
                     mirror=True, flipY=False, cover=False, blend="normal", occl=True, opac=(.85, 1.0)),
    "vidrio":   dict(scMin=1.0, scMax=1.8, rotMax=18, dx=.14, dy=.14, bx=.5, by=.5,
                     mirror=True, flipY=False, cover=True, blend="screen", occl=False, opac=(.45, .85)),
    "gotas":    dict(scMin=1.0, scMax=1.8, rotMax=18, dx=.14, dy=.14, bx=.5, by=.5,
                     mirror=True, flipY=False, cover=True, blend="normal", occl=False, opac=(.55, .95)),
    "destello": dict(scMin=.20, scMax=.95, rotMax=180, dx=.42, dy=.42, bx=.5, by=.4,
                     mirror=True, flipY=False, cover=False, blend="screen", occl=False, opac=(.5, .9)),
}


def _occ_cat(name: str) -> str:
    n = name.lower()
    if "lash" in n:
        return "pestañas"
    if "eyebrow" in n:
        return "cejas"
    if "glasses" in n or "lens" in n:
        return "anteojos"
    if "drop" in n:
        return "gotas"
    if "glint" in n or "destello" in n:
        return "destello"
    return "vidrio"          # glass_generated, texturas, etc.


_OCC_CACHE = None


def _load_occlusion_assets():
    """Lista de (nombre, RGBA, reglas). Las reglas parten de la categoría y se
    pisan con lo que el usuario guardó en src/assets/occlusion_spec.json (por archivo)."""
    global _OCC_CACHE
    if _OCC_CACHE is not None:
        return _OCC_CACHE
    from pathlib import Path
    import json
    root = Path(__file__).resolve().parents[1]
    d = root / "src" / "assets" / "occlusion"
    spec_assets = {}
    sp = root / "src" / "assets" / "occlusion_spec.json"
    if sp.exists():
        try:
            spec_assets = (json.loads(sp.read_text()).get("assets") or {})
        except Exception:  # noqa: BLE001
            spec_assets = {}
    out = []
    if d.is_dir():
        for f in sorted(d.iterdir()):
            im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
            elif im.shape[2] == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)   # el training trabaja en RGB
            rules = dict(_OCC_RULES[_occ_cat(f.name)])
            sv = spec_assets.get(f.name)
            if sv:                                        # pisar con lo guardado por el usuario
                for k in ("scMin", "scMax", "rotMax", "dx", "dy", "mirror", "cover", "flipY"):
                    if k in (sv.get("rules") or {}):
                        rules[k] = sv["rules"][k]
                b = sv.get("base") or {}
                if "x" in b:
                    rules["bx"] = b["x"]
                if "y" in b:
                    rules["by"] = b["y"]
            out.append((f.name, im, rules))
    _OCC_CACHE = out
    return out


def aug_occlusion(img: np.ndarray, mask: np.ndarray, force_asset=None) -> tuple:
    """Superpone UN asset de oclusión real (pestañas, cejas, un lente, vidrio,
    gotas o destello) con transformación aleatoria acotada por su categoría y por
    lo que guardaste en occlusion_spec.json (src/assets/occlusion/*). Los opacos
    (pestañas/cejas/lente) tapan la máscara donde cubren el iris. los transparentes
    (vidrio/gotas/destello) la dejan intacta. `force_asset`: nombre de archivo para
    forzar uno (para previews). Devuelve (img, mask)."""
    assets = _load_occlusion_assets()
    if not assets:
        # Public variant: synthetic eyelid geometry, no third-party image assets.
        return aug_eyelid_occlusion_v3(img, mask)
    if force_asset is not None:
        hit = [a for a in assets if a[0] == force_asset]
        name, a_rgba, R = (hit[0] if hit else assets[np.random.randint(len(assets))])
    else:
        name, a_rgba, R = assets[np.random.randint(len(assets))]
    H, W = img.shape[:2]
    ah, aw = a_rgba.shape[:2]
    ar = aw / ah

    sc = np.random.uniform(R["scMin"], R["scMax"])
    if R["cover"]:
        dw, dh = max(W, W * ar) * sc, max(W, W / ar) * sc
    else:
        t = sc * W
        dw, dh = (t, t / ar) if ar >= 1 else (t * ar, t)
    sx, sy = dw / aw, dh / ah
    if R["mirror"] and np.random.rand() < 0.5:
        sx = -sx
    by = R["by"]
    if R["flipY"] and np.random.rand() < 0.5:
        sy = -sy
        by = 1.0 - by
    cx = float(np.clip(R["bx"] + np.random.uniform(-R["dx"], R["dx"]), 0.02, 0.98)) * W
    cy = float(np.clip(by + np.random.uniform(-R["dy"], R["dy"]), 0.02, 0.98)) * H
    rot = np.deg2rad(np.random.uniform(-R["rotMax"], R["rotMax"]))
    cos, sin = np.cos(rot), np.sin(rot)
    Rm = np.array([[cos, -sin], [sin, cos]], np.float32) @ np.array([[sx, 0], [0, sy]], np.float32)
    off = np.array([cx, cy], np.float32) - Rm @ np.array([aw / 2.0, ah / 2.0], np.float32)
    M = np.hstack([Rm, off.reshape(2, 1)]).astype(np.float32)
    warped = cv2.warpAffine(a_rgba, M, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    wrgb = warped[:, :, :3].astype(np.float32)
    op = float(np.random.uniform(*R["opac"]))
    alpha = (warped[:, :, 3].astype(np.float32) / 255.0 * op)

    out = img.astype(np.float32)
    screen = (R["blend"] == "screen")
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        g = cv2.cvtColor(warped[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
        base = out if img.ndim == 2 else out[..., 0]
        blended = 255.0 - (255.0 - base) * (255.0 - g) / 255.0 if screen else g
        newv = base * (1 - alpha) + blended * alpha
        if img.ndim == 2:
            out = newv
        else:
            out[..., 0] = newv
    else:
        c = img.shape[2]
        wr = wrgb[..., :c]
        a3 = alpha[..., None]
        blended = 255.0 - (255.0 - out) * (255.0 - wr) / 255.0 if screen else wr
        out = out * (1 - a3) + blended * a3
    img_out = np.clip(out, 0, 255).astype(img.dtype)

    mask_out = mask.copy()
    if R["occl"]:
        solid = warped[:, :, 3] > 150                # zonas opacas del asset
        mask_out[solid & ((mask == 1) | (mask == 2))] = 0
    return img_out, mask_out


# ---------------------------------------------------------------------------
# v2: quality degradation augmentations
# ---------------------------------------------------------------------------

def aug_jpeg_compression(img: np.ndarray,
                         quality_range: tuple = (15, 60)) -> np.ndarray:
    """
    Re-encode as low-quality JPEG and decode → blocky / DCT-quantization
    artifacts. Simulates the typical webcam JPEG pipeline at low bitrate.

    quality 15 → very visible 8×8 block artifacts. 60 → mild.

    Reference: Yim & Park 2019 "DEEP CNN-BASED BLIND IMAGE QUALITY PREDICTOR"
    use the same trick for IQA augmentation.
    """
    quality = int(np.random.randint(quality_range[0], quality_range[1] + 1))
    img2d, sq = _squeeze(img)
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(".jpg", img2d, encode_param)
    if not ok:
        return img
    if img2d.ndim == 2:
        decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    else:
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return _unsqueeze(decoded, sq)


def aug_downscale_upscale(img: np.ndarray,
                          scale_range: tuple = (0.40, 0.80)) -> np.ndarray:
    """
    Downscale then upscale back to original size → loss of resolution / soft
    blurring. Simulates a low-resolution webcam stream or a heavily
    downsampled frame.

    scale 0.40 = 40% of original size before upscale (heavy quality loss);
    0.80 = 80% (mild). Sampled uniformly.
    """
    h, w = img.shape[:2]
    scale = float(np.random.uniform(*scale_range))
    img2d, sq = _squeeze(img)
    new_h = max(8, int(h * scale))
    new_w = max(8, int(w * scale))
    small = cv2.resize(img2d, (new_w, new_h), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return _unsqueeze(back, sq)


def aug_global_darken(img: np.ndarray,
                       factor_range: tuple = (0.25, 0.75)) -> np.ndarray:
    """
    Multiply the whole image by a darkening factor: separate from
    brightness/contrast jitter, more aggressive. Simulates underexposed
    captures (low light, backlit, etc.).

    factor 0.25 = very dark (25% intensity). 0.75 = mild dimming.
    """
    factor = float(np.random.uniform(*factor_range))
    return (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def aug_gaussian_noise_strong(img: np.ndarray,
                              std_range: tuple = (15, 50)) -> np.ndarray:
    """
    More aggressive gaussian noise than v1 (std 5-25 → 15-50). Simulates
    low-SNR webcam in poor lighting or with high ISO. Same additive model:
    zero-mean Gaussian per pixel.
    """
    std = float(np.random.uniform(*std_range))
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# v2 entry point
# ---------------------------------------------------------------------------

# =============================================================================
# v3: constrained variants (user-requested 2026-05-20)
#
#   Constraints learned from v2 failure modes:
#     1. Eyelid should ACTUALLY OCCLUDE the iris/pupil (not just float above
#        them). The v2 eyelid stops at ≤5% of the opposite edge so the iris
#        always stays clear: that doesn't teach the model anything about
#        real eyelid occlusion. v3 anchors the deepest point of the parabola
#        to touch the iris bbox edge (with random penetration into it).
#     2. Translation/zoom should keep ≥80% of the iris+pupil mask inside the
#        frame. Lost mask = lost supervision.
#     3. No reflect ever (already in v2). BORDER_CONSTANT only.
# =============================================================================


def _mask_fg(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of iris+pupil pixels (classes 1 and 2)."""
    return (mask == 1) | (mask == 2)


def _bbox_of(fg: np.ndarray):
    if not fg.any():
        return None
    ys, xs = np.where(fg)
    return ys.min(), xs.min(), ys.max() + 1, xs.max() + 1


def aug_eyelid_occlusion_v3(img: np.ndarray, mask: np.ndarray, force_side=None) -> tuple:
    """v3: corregido según el dibujo del usuario (src/assets/eyelid_spec.json).

    El párpado se modela como una parábola `y_cut(x)` que forma el borde de la
    abertura del ojo (más ABIERTA en el centro, se cierra hacia las esquinas):

      · SUPERIOR: viene desde ARRIBA. La curva está más ALTA (abierta) en el centro
        y BAJA hacia las esquinas → ocluye TODO lo que está ARRIBA de la curva.
      · INFERIOR: viene desde ABAJO. La curva está más BAJA (abierta) en el centro
        y SUBE hacia las esquinas → ocluye TODO lo que está ABAJO de la curva.

    Curvatura y penetración aleatorias dentro de un rango, pero la forma es siempre
    ésta. Garantiza que SIEMPRE se ocluya algo del iris (ajusta la penetración hasta
    tapar entre ~5% y ~70% del iris+pupila). si no hay iris, no toca.

    `force_side`: None (aleatorio), "upper" o "lower": para poder mostrar ambos.
    """
    h, w = img.shape[:2]
    if int((mask != 0).sum()) == 0:
        return img, mask                      # imagen de fondo (máscara vacía): nada que ocluir
    fg = _mask_fg(mask)
    bb = _bbox_of(fg)
    iris_pupil = (mask == 1) | (mask == 2)
    total = int(iris_pupil.sum())
    if bb is None or total < 10:
        return aug_eyelid_occlusion(img, mask)
    y0_bb, _, y1_bb, _ = bb
    bb_h = max(1, y1_bb - y0_bb)

    upper = (force_side == "upper") if force_side in ("upper", "lower") else (np.random.rand() < 0.5)
    cx = w / 2.0
    norm = ((np.arange(w) - cx) / (w / 2.0)) ** 2          # 0 en el centro → 1 en las esquinas
    amp = float(np.random.uniform(0.5, 1.3)) * bb_h        # cuánto baja/sube hacia las esquinas
    pen = float(np.random.uniform(0.20, 0.60)) * bb_h      # penetración del punto más profundo (esquinas) en el iris

    yy = np.indices((h, w))[0]
    erase = None
    y_cut = None
    for _ in range(8):
        if upper:
            y_edge = y0_bb + pen                            # esquinas entran pen dentro del iris (punto más ocluido)
            y_cut = (y_edge - amp) + amp * norm             # centro más alto (abierto), esquinas = y_edge
        else:
            y_edge = y1_bb - pen                            # esquinas suben pen dentro del iris
            y_cut = (y_edge + amp) - amp * norm             # centro más bajo (abierto), esquinas = y_edge
        y_cut = np.clip(np.round(y_cut).astype(np.int32), 0, h - 1)
        erase = (yy < y_cut[None, :]) if upper else (yy > y_cut[None, :])
        frac = float((erase & iris_pupil).sum()) / total
        if 0.05 <= frac <= 0.70:
            break
        pen += (0.15 * bb_h if frac < 0.05 else -0.15 * bb_h)
        pen = max(0.0, pen)

    edge_ys = np.clip(y_cut + (1 if upper else -1), 0, h - 1)
    edge_xs = np.arange(w)
    img_out = img.copy()
    mask_out = mask.copy()
    if img.ndim == 2:
        img_out[erase] = int(np.median(img[edge_ys, edge_xs]))
    elif img.shape[2] == 1:
        img_out[..., 0][erase] = int(np.median(img[edge_ys, edge_xs, 0]))
    else:
        img_out[erase] = np.median(img[edge_ys, edge_xs, :], axis=0).astype(img.dtype)
    mask_out[erase] = 0
    return img_out, mask_out


def _mask_inside_pct(mask_before: np.ndarray, mask_after: np.ndarray) -> float:
    """Fraction of iris+pupil pixels preserved (after / before)."""
    fg_before = (mask_before == 1).sum() + (mask_before == 2).sum()
    if fg_before == 0:
        return 1.0
    fg_after = (mask_after == 1).sum() + (mask_after == 2).sum()
    return float(fg_after) / float(fg_before)


def aug_translation_v3(img: np.ndarray, mask: np.ndarray,
                       max_shift_frac: float = 0.45,
                       min_keep: float = 0.80,
                       max_retries: int = 8) -> tuple:
    """v3 translation: garantiza que al menos `min_keep` del iris+pupila queda
    adentro del frame después del shift. Si la translación pierde demasiado,
    reduce la magnitud hasta que cumpla. Sin mirror/reflect.
    """
    fg_before = (mask == 1).sum() + (mask == 2).sum()
    if fg_before == 0:
        return aug_translation_v2(img, mask, max_shift_frac)

    factor = 1.0
    for _ in range(max_retries):
        out_img, out_mask = aug_translation_v2(img, mask, max_shift_frac=max_shift_frac * factor)
        if _mask_inside_pct(mask, out_mask) >= min_keep:
            return out_img, out_mask
        factor *= 0.6
    # Last resort: identity (no translation)
    return img, mask


def aug_zoom_v3(img: np.ndarray, mask: np.ndarray,
                scale_range: tuple = (0.55, 1.30),
                min_keep: float = 0.80,
                max_retries: int = 6) -> tuple:
    """v3 zoom: garantiza ≥`min_keep` del iris+pupila adentro. Re-muestrea
    scale si pierde demasiado. Zoom-in (>1) puede recortar. reducimos el upper
    cap si pierde.
    """
    fg_before = (mask == 1).sum() + (mask == 2).sum()
    if fg_before == 0:
        return aug_zoom(img, mask, scale_range)

    lo, hi = scale_range
    for _ in range(max_retries):
        out_img, out_mask = aug_zoom(img, mask, scale_range=(lo, hi))
        if _mask_inside_pct(mask, out_mask) >= min_keep:
            return out_img, out_mask
        # Tighten upper bound (zoom-in is the destructive direction)
        hi = max(1.0, (1.0 + hi) / 2.0)
    return img, mask


def aug_rotation_v3(img: np.ndarray, mask: np.ndarray,
                    max_angle: float = 25.0,
                    min_keep: float = 0.80,
                    max_retries: int = 4) -> tuple:
    """v3 rotation: garantiza ≥`min_keep` del iris+pupila adentro.
    Rotación grande puede expulsar parte del mask por las esquinas.
    """
    fg_before = (mask == 1).sum() + (mask == 2).sum()
    if fg_before == 0:
        return aug_rotation(img, mask, max_angle)

    cur_angle = max_angle
    for _ in range(max_retries):
        out_img, out_mask = aug_rotation(img, mask, max_angle=cur_angle)
        if _mask_inside_pct(mask, out_mask) >= min_keep:
            return out_img, out_mask
        cur_angle *= 0.6
    return img, mask


def apply_augmentations_v3(img: np.ndarray, mask: np.ndarray) -> tuple:
    """v3 augmentation pipeline: v2 estructura + constraints:

      - flip (50%)
      - ONE geometric: translation_v3 (60%) | zoom_v3 (30%) | both (10%)
        → garantiza ≥80% del iris+pupila queda en el frame
      - rotation_v3 (35%) → idem
      - ONE structural (40%): eyelid_v3 (50%) | vertical (15%) | pupil_resize (35%)
        → eyelid_v3 hace que la curva del párpado TOQUE el bbox del iris
      - ONE appearance (60%): 10 técnicas: sin cambios vs v2

    Diferencia con v2: ya no perdemos máscara con translation/zoom agresivos,
    y el párpado simulado realmente ocluye el iris.
    """
    img, mask = aug_flip(img, mask)

    r = np.random.rand()
    if r < 0.60:
        img, mask = aug_translation_v3(img, mask)
    elif r < 0.90:
        img, mask = aug_zoom_v3(img, mask)
    else:
        img, mask = aug_translation_v3(img, mask)
        img, mask = aug_zoom_v3(img, mask)

    if np.random.rand() < 0.35:
        img, mask = aug_rotation_v3(img, mask)

    if np.random.rand() < 0.40:
        struct_pool = [aug_eyelid_occlusion_v3, aug_vertical_occlusion, aug_pupil_resize, aug_erase_iris]
        weights = [0.46, 0.14, 0.32, 0.08]   # erase_iris = poco común
        idx = int(np.random.choice(len(struct_pool), p=weights))
        img, mask = struct_pool[idx](img, mask)

    if np.random.rand() < 0.22:              # oclusión con asset real (pestañas/lente/vidrio/gotas/destello)
        img, mask = aug_occlusion(img, mask)

    if np.random.rand() < 0.60:
        appear_pool = [
            (aug_gaussian_blur,          False),
            (aug_random_lines,           False),
            (aug_starburst,              False),
            (aug_brightness_contrast,    False),
            (aug_global_darken,          False),
            (aug_skin_tone,              False),
            (aug_iris_darken,            True),
            (aug_iris_color,             True),
            (aug_jpeg_compression,       False),
            (aug_downscale_upscale,      False),
            (aug_gaussian_noise_strong,  False),
        ]
        idx = np.random.randint(len(appear_pool))
        fn, needs_mask = appear_pool[idx]
        img = fn(img, mask) if needs_mask else fn(img)

    return img, mask
