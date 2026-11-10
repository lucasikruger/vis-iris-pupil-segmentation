"""Entrenamiento ResU-Net final. evaluación LODO con EMA de última época."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import EyeDataset
from src.ema import EMA
from src.losses import SegmentationLoss
from src.metrics import compute_metrics, aggregate_metrics
from src.models import build_model
from src.splits import partition

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metrics = []
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        metrics.append(compute_metrics(pred, y))
    return aggregate_metrics(metrics)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='src/configs/final.yaml')
    p.add_argument('--data-root'); p.add_argument('--csv'); p.add_argument('--fold')
    p.add_argument('--epochs', type=int); p.add_argument('--limit', type=int)
    p.add_argument('--output'); p.add_argument('--device', default='cpu')
    a = p.parse_args()
    c = yaml.safe_load(Path(a.config).read_text())
    for arg, key in [('data_root','data_root'),('csv','input_csv'),('fold','fold'),('epochs','epochs'),('output','output')]:
        if getattr(a,arg) is not None: c[key] = getattr(a,arg)
    from src.augmentations import _load_occlusion_assets
    occlusions = _load_occlusion_assets()
    c['occlusion_assets'] = [name for name, _, _ in occlusions]
    c['occlusion_variant'] = 'external-assets' if occlusions else 'procedural'
    if not occlusions:
        print('AVISO: variante pública sin assets externos; se usan oclusiones procedurales. '
              'La tesis usó siete assets. Ver README.md.', flush=True)
    elif len(occlusions) != 7:
        print('AVISO: assets parciales; la receta de tesis usó siete archivos.', flush=True)
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    torch.set_num_threads(4)
    df = partition(pd.read_csv(c['input_csv']), c['fold'], c['seed'])
    out = Path(c['output']); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/'partitions.csv', index=False)
    loaders = {}
    for split in ['train','val','test']:
        rows = df[df.split == split]
        if a.limit: rows = rows.head(a.limit)
        if rows.empty: continue
        ds = EyeDataset(rows, c['data_root'], grayscale=False, augment=split=='train',
                        **{k:c[k] for k in ['img_size','aug_version','online_crop','framing_policy','crop_target_iris_pct','crop_jitter_pct','crop_zoom_jitter']})
        loaders[split] = DataLoader(ds, batch_size=c['batch_size'], shuffle=split=='train',
                                   num_workers=c['num_workers'], drop_last=False)
    if not {'train','val'} <= loaders.keys(): raise ValueError('Se requieren train y val')
    model = build_model().to(a.device)
    loss_fn = SegmentationLoss().to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c['epochs'], eta_min=c['lr']/100)
    ema = EMA(model, c['ema_decay']); best = -1; history = []
    for epoch in range(1,c['epochs']+1):
        model.train(); total = 0; count = 0
        for x,y in loaders['train']:
            x,y=x.to(a.device),y.to(a.device); opt.zero_grad()
            loss=loss_fn(model(x),y); loss.backward(); opt.step(); ema.update(model)
            total+=loss.item()*len(x); count+=len(x)
        sch.step(); backup=ema.apply_to(model)
        metrics=evaluate(model,loaders['val'],a.device)
        ckpt=dict(model_state=model.state_dict(),model_name=c['model'],img_size=96,
                  in_channels=3,num_classes=3,epoch=epoch,uses_ema=True,config=c)
        torch.save(ckpt,out/'last.pt')
        if metrics['iou_mean']>best:
            best=metrics['iou_mean']; torch.save(ckpt,out/'best.pt')
        record=dict(epoch=epoch,train_loss=total/count,val=metrics)
        history.append(record); print(json.dumps(record),flush=True)
        ema.restore(model,backup)
    if 'test' in loaders:
        ema.apply_to(model)
        (out/'test.json').write_text(json.dumps(evaluate(model,loaders['test'],a.device),indent=2))
    (out/'history.json').write_text(json.dumps(history,indent=2))

if __name__ == '__main__': main()
