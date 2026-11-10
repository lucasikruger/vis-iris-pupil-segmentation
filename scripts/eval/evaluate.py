"""Evaluación webcam: protocolo de la tesis, margen 12 %, CLAHE OpenCV."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models import build_model
from src.metrics import compute_metrics, aggregate_metrics
from src.webcam import RecorteConPadding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='dataset-package')
    p.add_argument('--weights',default='weights/model.pt')
    p.add_argument('--limit',type=int)
    p.add_argument('--output',default='docs/results/webcam.json')
    a=p.parse_args();torch.set_num_threads(4)
    root=Path(a.data_root)
    df=pd.read_csv(root/'webcam_manual_mobius-format/data.csv')
    df=df[df.rel_mask_path.map(lambda x:(root/x).is_file())].reset_index(drop=True)
    if a.limit:df=df.head(a.limit)
    ds=RecorteConPadding(df,0.12,root)
    model=build_model();model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True)['model_state']);model.eval()
    batches=[]; distances=[];missing=0;offset=0
    with torch.no_grad():
        for x,y in DataLoader(ds,batch_size=16):
            pred=model(x).argmax(1);batches.append(compute_metrics(pred,y))
            for pp,yy in zip(pred.numpy(),y.numpy()):
                row=df.iloc[offset]; offset+=1
                ip,it=pp>0,yy>0
                missing+=int(ip.sum()<5)
                if ip.any() and it.any():
                    side=(ds.caja(row.stem) or (0,0,512))[2]
                    scale=(row.crop_x1-row.crop_x0)*side/512/96
                    distances.append(float(np.linalg.norm(np.array(np.nonzero(ip)).mean(1)-np.array(np.nonzero(it)).mean(1))*scale))
    result=dict(n=len(df),subjects=int(df.subject.nunique()),padding=0.12,
                aggregation='mean of batch IoU, batch_size=16, original thesis protocol',
                **aggregate_metrics(batches),centdist_orig_px=float(np.mean(distances)),sin_iris_pct=100*missing/len(df))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
