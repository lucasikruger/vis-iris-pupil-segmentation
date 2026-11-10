"""Recreate public gallery assets using only already cropped eye images."""
import argparse,json
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=Path('dataset-package'));a=p.parse_args()
root=Path(__file__).resolve().parents[2]
for entry in json.loads((root/'demos/web/public/examples/examples.json').read_text()):
    if not entry.get('src'):continue
    assert entry['eye_crop']
    image=Image.open(a.data_root/entry['source']).convert('RGB')
    if 'crop_xywh' in entry:
        x,y,w,h=entry['crop_xywh'];image=image.crop((x,y,x+w,y+h))
    for field,size in [('src',entry.get('image_max_size',256)),('thumbnail',120)]:
        out=image.copy();out.thumbnail((size,size));out.save(root/'demos/web/public'/entry[field].lstrip('/'),lossless=True)
