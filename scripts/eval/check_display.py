"""Compare web display interpolation with OpenCV and quantify 207 cleanup areas."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from PIL import Image,ImageDraw
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.postprocess import largest,postprocess
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=Path('runs/preprocess-parity'));p.add_argument('--predictions',type=Path,default=Path('runs/display-check'));a=p.parse_args();out=Path('reports/render');out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((a.predictions/'manifest.json').read_text());rows=[];max_diff=0;changed=0
for row in manifest['images']:
 i=row['index'];raw=np.fromfile(a.predictions/f'{i}.raw',np.uint8).reshape(96,96);clean=np.fromfile(a.predictions/f'{i}.clean',np.uint8).reshape(96,96)
 assert np.array_equal(postprocess(raw),clean)
 before_close=binary_fill_holes(largest(raw>0));closed=cv2.morphologyEx(before_close.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))>0
 r={'stem':row['stem'],'index':i,'disc_raw':int((raw>0).sum()),'disc_clean':int((clean>0).sum()),'iris_raw':int((raw==1).sum()),'iris_clean':int((clean==1).sum()),'pupil_raw':int((raw==2).sum()),'pupil_clean':int((clean==2).sum()),'closing_added_disc_pixels':int(closed.sum()-before_close.sum())};rows.append(r)
 if i<3:
  probs=np.fromfile(a.predictions/f'{i}.probs',np.float32).reshape(3,96,96)
  expected=np.stack([cv2.resize(c,(257,193),interpolation=cv2.INTER_LINEAR) for c in probs])
  actual=np.fromfile(a.predictions/f'{i}.resized',np.float32).reshape(3,193,257)
  max_diff=max(max_diff,float(np.abs(expected-actual).max()));changed+=int((expected.argmax(0)!=actual.argmax(0)).sum())
assert max_diff<1e-6,max_diff
summary={}
for name in ['disc','iris','pupil']:
 raw=np.array([r[name+'_raw'] for r in rows]);clean=np.array([r[name+'_clean'] for r in rows]);delta=clean-raw
 summary[name]={'raw_total_px':int(raw.sum()),'clean_total_px':int(clean.sum()),'delta_total_px':int(delta.sum()),'delta_pct_total':float(100*delta.sum()/raw.sum()),'mean_delta_px_per_crop':float(delta.mean()),'median_delta_px':float(np.median(delta)),'min_delta_px':int(delta.min()),'max_delta_px':int(delta.max()),'increased_crops':int((delta>0).sum()),'decreased_crops':int((delta<0).sum()),'unchanged_crops':int((delta==0).sum())}
added=np.array([r['closing_added_disc_pixels'] for r in rows]);report={'n':len(rows),'definitions':{'disc':'iris + pupil, labels > 0','iris':'iris ring, label 1','pupil':'label 2'},'areas_at_native_96':summary,'isolated_closing_5x5':{'added_total_px':int(added.sum()),'mean_added_px':float(added.mean()),'max_added_px':int(added.max()),'affected_crops':int((added>0).sum())},'opencv_inter_linear_parity':{'tested_maps':3,'output_size':[257,193],'max_probability_error':max_diff,'argmax_disagreements':changed},'web_onnx_miou_96_unchanged':manifest['miou_96'],'official_pytorch_miou_96_reference':json.loads(Path('docs/results/webcam.json').read_text())['iou_mean'],'images':rows}
(out/'area-and-alignment.json').write_text(json.dumps(report,indent=2)+'\n')
for i in manifest['selected']:
 row=rows[i];image=np.fromfile(a.inputs/f'{i}.rgb',np.uint8).reshape(96,96,3);base=cv2.resize(image,(384,384),interpolation=cv2.INTER_LINEAR)
 sheet=Image.new('RGB',(4*384,444),'#161a20');draw=ImageDraw.Draw(sheet)
 for j,(kind,version) in enumerate([('raw','before'),('raw','after'),('clean','before'),('clean','after')]):
  side=96 if version=='before' else 384;rgba=np.fromfile(a.predictions/f'{i}.{kind}.{version}',np.uint8).reshape(side,side,4)
  if version=='before':rgba=cv2.resize(rgba,(384,384),interpolation=cv2.INTER_NEAREST)
  alpha=rgba[:,:,3:4].astype(float)/255*.55;overlay=np.rint(base*(1-alpha)+rgba[:,:,:3]*alpha).astype(np.uint8)
  sheet.paste(Image.fromarray(overlay),(j*384,60));draw.text((j*384+10,10),f'{kind}: '+('ANTES (etiquetas NN)' if version=='before' else 'DESPUES (prob. bilineal)'),fill='white')
  draw.text((j*384+10,30),row['stem'],fill='white')
 sheet.save(out/f'comparison-{i}.webp',quality=94)
print(json.dumps({k:v for k,v in report.items() if k!='images'},indent=2))
