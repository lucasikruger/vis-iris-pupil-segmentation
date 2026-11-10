import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import cv2
import numpy as np
from src.nir2vis.models import UNetGenerator

def load_generator(ckpt_path: Path, device: str, vis_mode: str = "gray") -> tuple[UNetGenerator, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    out_nc = 3 if vis_mode == "rgb" else 1
    G = UNetGenerator(
        input_nc=1, output_nc=out_nc,
        num_downs=saved.get("num_downs", 8),
        ngf=saved.get("ngf", 64),
    ).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    return G, saved.get("img_size", 256)

@torch.no_grad()
def translate(G: UNetGenerator, gray: np.ndarray, img_size: int, device: str,
              vis_mode: str = "gray") -> np.ndarray:
    """Returns (H,W) gray or (H,W,3) BGR depending on vis_mode."""
    h, w = gray.shape
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0).to(device) / 127.5 - 1.0
    fake = G(t)
    fake = ((fake.clamp(-1, 1) + 1) * 127.5).cpu().numpy()  # (1,C,H,W)
    if vis_mode == "rgb":
        arr_rgb = fake[0].astype(np.uint8).transpose(1, 2, 0)
        arr_rgb = cv2.resize(arr_rgb, (w, h), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    arr = fake[0, 0].astype(np.uint8)
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_CUBIC)
