"""Check crop coverage, coordinate integrity and the retained PolyU/CASIA masks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.crop_metadata import load_boxes, relative_path


def check(root):
    root=Path(root)
    df=pd.read_csv(ROOT/'data/partitions.csv')
    report={}
    sources={'fei':'fei_manual','casia':'casia_pix2pix_rgb','mobius':'mobius',
             'polyu':'polyu_zoomed','ubiris':'ubiris_v2_with_pupils_zoomed'}
    for source in [*sources,'webcam','white_eyes_train','white_eyes_test']:
        records=load_boxes(root,source)
        if source in sources:
            expected={Path(p).stem for p in df[df.dataset==sources[source]].rel_image_path}
        else:
            base={'webcam':'webcam_manual_mobius-format','white_eyes_train':'white_eyes_bg','white_eyes_test':'white_eyes_bg_test'}[source]
            expected={p.stem for p in (root/base/'Images').glob('*.png')}
        if set(records)!=expected:
            raise ValueError(f'{source}: missing {sorted(expected-set(records))}, extra {sorted(set(records)-expected)}')
        if source == 'fei':
            legacy = json.loads((root/'annotations/fei_bboxes.json').read_text())
            if set(legacy) != expected:
                raise ValueError('FEI legacy boxes do not cover the same 210 crops')
        for stem,r in records.items():
            relative_path(root,r['source_image'])
            x0,y0,x1,y1=r['bbox_xyxy'];w,h=r['source_size']
            if not (all(type(v) is int for v in [x0,y0,x1,y1]) and 0<=x0<x1<=w and 0<=y0<y1<=h):
                raise ValueError(f'{source}/{stem}: invalid box')
            if source=='webcam':
                if len(r['mediapipe_contour_xy'])!=16:raise ValueError(stem)
                ox,oy=r['raw_crop_bbox_xyxy'][:2]
                assert r['mediapipe_contour_xy']==[[u+ox,v+oy] for u,v in r['raw_contour_crop_xy']]
            if source=='mobius':
                a,b,c,d=r['evaluation_bbox_xyxy']
                if not 0<=a<c<=w or not 0<=b<d<=h:raise ValueError(stem)
        report[source]={'records':len(records),'missing':0,'invalid_boxes':0}
    manifest=json.loads((ROOT/'data/prepared_sha256.json').read_text())
    for source in ['polyu_zoomed','casia_pix2pix_rgb']:
        paths=df[df.dataset==source].rel_mask_path.tolist()
        for rel in paths:
            if hashlib.sha256(relative_path(root,rel).read_bytes()).hexdigest()!=manifest[rel]:
                raise ValueError(f'Mask changed or missing: {rel}')
        report[source+'_masks']={'retained':len(paths),'sha256_mismatches':0}
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',default='dataset-package')
    parser.add_argument('--output',default='reports/crop-boxes.json')
    args=parser.parse_args();report=check(args.package)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
