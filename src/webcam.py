import csv
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .augmentations import preprocess
from .mask_formats import get_converter
IMG_SIZE=96
LADO=512

class RecorteConPadding(Dataset):
    """Rehace el recorte con otro padding, partiendo del contorno crudo del detector.

    Los archivos en disco estan recortados con padding 0,40. La caja con otro padding se
    calcula sobre la captura original, donde vive el contorno que devolvio MediaPipe, y se
    traduce al marco de 512 con el afin que quedo guardado en bboxes.json. Como la caja mas
    chica cae dentro de la grande, alcanza con recortar y reescalar lo ya guardado.
    """

    def __init__(self, df, pad, data_root):
        self.root = Path(data_root)
        self.dst = self.root / "webcam_manual_mobius-format"
        self.df, self.pad = df.reset_index(drop=True), pad
        canonical=self.root/'annotations/webcam_boxes.json'
        if canonical.exists():
            from .crop_metadata import load_boxes
            records=load_boxes(self.root,'webcam')
            self.bb={stem:{'crop_bbox_in_orig':r['bbox_xyxy']} for stem,r in records.items()}
            self.crudo={stem:{'contorno':json.dumps(r['raw_contour_crop_xy']),
                              'x':r['raw_crop_bbox_xyxy'][0],'y':r['raw_crop_bbox_xyxy'][1]}
                        for stem,r in records.items()}
        else:
            self.bb = json.loads((self.dst / "bboxes.json").read_text())
            self.crudo = {Path(r["recorte"]).stem: r
                          for r in csv.DictReader((self.dst / "crops.csv").open())}

    def __len__(self):
        return len(self.df)

    def caja(self, stem):
        r, meta = self.crudo.get(stem), self.bb.get(stem)
        if r is None or meta is None:
            return None
        pts = np.array(json.loads(r["contorno"])) + np.array([int(r["x"]), int(r["y"])])
        x0, y0, x1, y1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
        s = max(x1 - x0, y1 - y0) * (1 + 2 * self.pad)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ox, oy, ox1, _ = meta["crop_bbox_in_orig"]
        k = LADO / (ox1 - ox)                       # original -> marco guardado
        u0, v0, lado = (cx - s / 2 - ox) * k, (cy - s / 2 - oy) * k, s * k
        u0, v0 = int(max(0, round(u0))), int(max(0, round(v0)))
        lado = int(min(round(lado), LADO - u0, LADO - v0))
        return (u0, v0, lado) if lado >= 16 else None

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2.imread(str(self.root / row["rel_image_path"]), cv2.IMREAD_COLOR)
        mask = get_converter(row["dataset_format"])(
            cv2.imread(str(self.root / row["rel_mask_path"]), cv2.IMREAD_COLOR))
        c = self.caja(row["stem"])
        if c is not None:
            u, v, l = c
            img, mask = img[v:v + l, u:u + l], mask[v:v + l, u:u + l]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_NEAREST).astype(np.int64)
        img = preprocess(img, grayscale=False)
        return (torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0,
                torch.from_numpy(mask).long())
