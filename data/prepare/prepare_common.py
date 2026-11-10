"""Reconstrucción a partir de originales autorizados y anotaciones separadas."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from crop_recipes import tight_eyelid_crop,resize_to

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.crop_metadata import load_boxes, relative_path, crop_from_record

NAMES={'polyu':'polyu_zoomed','ubiris':'ubiris_v2_with_pupils_zoomed','fei':'fei_manual',
       'casia':'casia_pix2pix_rgb','tayed':'tayed_pix2pix_rgb','miche':'miche','mobius':'mobius',
       'closed_eyes':'closed_eyes_bg','white_eyes':'white_eyes_bg','webcam':'webcam_manual'}

def read(path,flag=cv2.IMREAD_COLOR):
    im=cv2.imread(str(path),flag)
    if im is None:raise FileNotFoundError(path)
    return im

def jpeg_roundtrip(im,quality=95):
    ok,data=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,quality])
    if not ok:raise RuntimeError('JPEG encode')
    return cv2.imdecode(data,cv2.IMREAD_COLOR)

def find(root,name):
    matches=list(root.rglob(name))
    if len(matches)!=1:raise ValueError(f'Se esperaba un único {name}; encontrados {len(matches)}')
    return matches[0]

def main(source):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True,help='Originales autorizados; ver data/DATA.md')
    p.add_argument('--annotations',type=Path,default=Path('dataset-package'))
    p.add_argument('--output',type=Path,default=Path('data/mobius_format'))
    p.add_argument('--limit',type=int)
    p.add_argument('--masks',type=Path,help='Máscaras de terceros obtenidas con su permiso')
    p.add_argument('--generator',type=Path,help='Checkpoint RGB de pix2pix para CASIA/TEyeD')
    p.add_argument('--nir-only',action='store_true',help='CASIA: reconstruye recortes NIR exactos, sin pix2pix')
    p.add_argument('--stage',choices=['pool','evaluation'],default='pool',help='MOBIUS: copia del pool o recorte de evaluación 96×96')
    a=p.parse_args()
    if a.nir_only and source != 'casia':raise ValueError('--nir-only es solo para CASIA')
    if a.stage != 'pool' and source != 'mobius':raise ValueError('--stage evaluation es solo para MOBIUS')
    if a.raw.resolve()==a.output.resolve():raise ValueError('Entrada y salida deben ser distintas')
    if source in ['white_eyes','webcam']:
        from prepare_owned import prepare_owned
        prepare_owned(source, a)
        return
    df=pd.read_csv(ROOT/'data/partitions.csv');df=df[df.dataset==NAMES[source]]
    if a.limit:df=df.head(a.limit)
    hashes=json.loads((ROOT/'data/prepared_sha256.json').read_text())
    mismatches=[]
    if source in ['fei','casia','mobius','polyu','ubiris']:boxes=load_boxes(a.annotations, source)
    if source in ['casia','tayed'] and not a.nir_only:
        if a.generator is None:raise ValueError('Se requiere --generator; ver pendientes de reconstrucción NIR')
        import torch
        torch.set_num_threads(4)
        from nir_translate import load_generator,translate
        generator,size=load_generator(a.generator,'cpu','rgb')
    for _,row in df.iterrows():
        stem=Path(row.rel_image_path).stem
        image_dest=a.output/row.rel_image_path;mask_dest=a.output/row.rel_mask_path
        image_dest.parent.mkdir(parents=True,exist_ok=True);mask_dest.parent.mkdir(parents=True,exist_ok=True)
        quality=95
        if source=='mobius':
            meta=boxes[stem]
            raw=read(relative_path(a.raw,meta['source_image']))
            crop_from_record(raw,meta)  # Validate that pool files preserve the full frame.
            if a.stage=='evaluation':
                im=crop_from_record(raw,meta,box_key='evaluation_bbox_xyxy',size_key='evaluation_output_size',interpolation='LINEAR')
                # Segmentation labels must use nearest-neighbor interpolation.
                x0,y0,x1,y1=meta['evaluation_bbox_xyxy']
                mask=cv2.resize(read(a.raw/row.rel_mask_path)[y0:y1,x0:x1],(96,96),interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(image_dest),im,[cv2.IMWRITE_JPEG_QUALITY,95]);cv2.imwrite(str(mask_dest),mask)
                continue
            shutil.copy2(a.raw/row.rel_image_path,image_dest)
            shutil.copy2(a.raw/row.rel_mask_path,mask_dest)
        elif source=='fei':
            meta=boxes[stem];raw=read(find(a.raw,meta['source_image']))
            im=crop_from_record(raw,meta)
            mask=read(a.annotations/row.rel_mask_path)
        elif source=='polyu':
            _,subject,side,number=stem.split('_')
            raw=read(find(a.raw,f'{subject}_{side}_VIS_{number}.tiff'))
            mask=read(a.annotations/'annotations/polyu_original'/f'{stem}.png')
            meta=boxes[stem]
            im=crop_from_record(jpeg_roundtrip(raw),meta)
            x0,y0,x1,y1=meta['bbox_xyxy'];mask=cv2.resize(mask[y0:y1,x0:x1],tuple(meta['output_size']),interpolation=cv2.INTER_NEAREST)
        elif source=='ubiris':
            if a.masks is None:raise ValueError('UBIRIS requiere --masks con iris original (blanco=iris)')
            raw=read(find(a.raw,stem+'.tiff'));iris=read(find(a.masks,'OperatorA_'+stem+'.tiff'),cv2.IMREAD_GRAYSCALE)>127
            pupil=read(a.annotations/'annotations/ubiris_pupil_original'/f'{stem}.png',cv2.IMREAD_GRAYSCALE)>127
            union=binary_fill_holes(iris)|pupil
            mask=np.zeros((*iris.shape,3),dtype=np.uint8);mask[union,1]=255;mask[pupil,0]=255;mask[~union,2]=255
            meta=boxes[stem]
            im=crop_from_record(jpeg_roundtrip(raw),meta)
            x0,y0,x1,y1=meta['bbox_xyxy'];mask=cv2.resize(mask[y0:y1,x0:x1],tuple(meta['output_size']),interpolation=cv2.INTER_NEAREST)
        elif source=='miche':
            # The authors' Wang eye crops and RGB reference masks, before thesis eyelid clipping.
            if a.masks is None:raise ValueError('MICHE requiere GT autorizado --masks, formato MOBIUS a 512×512')
            im=read(find(a.raw,stem+'.jpg'));mask=read(find(a.masks,stem+'.png'))
            im=cv2.resize(im,(512,512));mask=cv2.resize(mask,(512,512),interpolation=cv2.INTER_NEAREST)
            meta=json.loads((a.annotations/'annotations/miche_eyelid'/f'{stem}.json').read_text())
            decision=json.loads((ROOT/'data/recipes/miche_decisions.json').read_text()).get(stem,'keep')
            poly=meta.get('polygon_out_size') if decision!='no_eyelid' else None
            if poly is not None:
                lid=np.zeros((512,512),np.uint8);cv2.fillPoly(lid,[np.array(poly,dtype=np.int32)],255)
                mask[lid==0]=0
            im,mask,poly=tight_eyelid_crop(im,mask,poly,0.10)
            im,mask,_=resize_to(im,mask,poly,512);quality=92
        elif source in ['casia','tayed']:
            if source=='casia':
                meta=boxes[stem]
                nir=crop_from_record(read(relative_path(a.raw,meta['source_image'])),meta)
                if a.nir_only:
                    dest=a.output/'casia_distance_hybrid_mobius-format/Images'/f'{stem}.png'
                    dest.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(dest),nir)
                    maskdest=a.output/'casia_distance_hybrid_mobius-format/Masks'/f'{stem}.png'
                    maskdest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.annotations/row.rel_mask_path,maskdest)
                    if hashlib.sha256(dest.read_bytes()).hexdigest()!=meta['crop_sha256']:mismatches.append(str(dest.relative_to(a.output)))
                    continue
                gray=cv2.cvtColor(nir,cv2.COLOR_BGR2GRAY)
            else:
                files=[p for p in a.raw.rglob(stem+'.*') if p.suffix.lower() in ['.png','.jpg']]
                if len(files)!=1:raise ValueError(f'Falta recorte NIR inequívoco: {stem}')
                gray=read(files[0],cv2.IMREAD_GRAYSCALE)
            im=translate(generator,gray,size,'cpu','rgb');quality=92
            if source=='casia':mask=read(a.annotations/row.rel_mask_path)
            else:
                if a.masks is None:raise ValueError('TEyeD requiere máscara nativa convertida --masks (MOBIUS)')
                mask=read(find(a.masks,stem+'.png'))
        elif source=='closed_eyes':
            # Exact curated selection: look up decoded pixels, independent of source filename.
            expected=json.loads((ROOT/'data/recipes/closed_pixels.json').read_text())
            if not hasattr(main,'closed_index'):
                index={}
                for f in a.raw.rglob('*'):
                    if f.suffix.lower() not in ['.png','.jpg','.jpeg','.bmp']:continue
                    im0=read(f);index[hashlib.sha256(im0.tobytes()).hexdigest()]=f
                main.closed_index=index
            im=read(main.closed_index[expected[stem]])
            mask=np.zeros_like(im)
        if source!='mobius':
            opts=[cv2.IMWRITE_JPEG_QUALITY,quality] if image_dest.suffix=='.jpg' else []
            if not cv2.imwrite(str(image_dest),im,opts) or not cv2.imwrite(str(mask_dest),mask):raise OSError('No se pudo escribir')
        for dest,rel in [(image_dest,row.rel_image_path),(mask_dest,row.rel_mask_path)]:
            if hashlib.sha256(dest.read_bytes()).hexdigest()!=hashes[rel]:mismatches.append(rel)
    report={'source':source,'n':len(df),'stage':('nir' if a.nir_only else a.stage),'sha256_mismatches':mismatches}
    (a.output/f'prepare_{source}_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if mismatches:raise SystemExit('Hay diferencias con la exportación de investigación; revisar reporte antes de entrenar')
