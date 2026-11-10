"""Export the 207 exact 96px webcam inputs and Python references for Node parity."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.webcam import RecorteConPadding
from src.augmentations import preprocess
p=argparse.ArgumentParser();p.add_argument('--data-root',default='dataset-package');p.add_argument('--output',default='runs/preprocess-parity');a=p.parse_args()
root=Path(a.data_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(root/'webcam_manual_mobius-format/data.csv');df=df[df.rel_mask_path.map(lambda x:(root/x).is_file())].reset_index(drop=True)
ds=RecorteConPadding(df,.12,root);manifest=[]
for i,row in df.iterrows():
    image=cv2.imread(str(root/row.rel_image_path));box=ds.caja(row.stem)
    if box:
        u,v,l=box;image=image[v:v+l,u:u+l]
    image=cv2.resize(image,(96,96),interpolation=cv2.INTER_LINEAR)
    cv2.cvtColor(image,cv2.COLOR_BGR2RGB).tofile(out/f'{i}.rgb')
    preprocess(image,grayscale=False).tofile(out/f'{i}.expected')
    ds[i][1].numpy().astype(np.uint8).tofile(out/f'{i}.mask')
    manifest.append(dict(index=i,stem=row.stem))
(out/'manifest.json').write_text(json.dumps(dict(opencv=cv2.__version__,size=96,images=manifest),indent=2))
print(f'Exported {len(manifest)} crops to {out}')
