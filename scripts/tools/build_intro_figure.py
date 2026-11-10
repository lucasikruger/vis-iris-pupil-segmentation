"""Versión redistribuible del diseño de intro_seg_fig.py de la tesis.
Arriba: recortes propios; abajo: los mismos con anotaciones de referencia.
No usa predicciones, capturas completas ni imágenes de bases de terceros.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

IRIS, PUPIL = (60, 220, 90), (70, 120, 255)
EXAMPLES = ['frontal', 'lateral', 'anteojos', 'poca-luz', 'reflejo-moderado']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('dataset-package'))
    p.add_argument('--output', type=Path, default=Path('docs/img/datos-intro.png'))
    args = p.parse_args()
    entries = json.loads(Path('demos/web/public/examples/examples.json').read_text())
    size, gap, padding = 300, 8, 14
    figure = Image.new('RGB', (2 * padding + len(EXAMPLES) * size + (len(EXAMPLES)-1) * gap,
                               2 * padding + 2 * size + gap), 'white')
    records = []
    for i, name in enumerate(EXAMPLES):
        entry = next(e for e in entries if e['id'] == name)
        image_path = Path(entry['source'])
        assert image_path.parts[:2] == ('webcam_manual_mobius-format', 'Images')
        mask_path = Path(*image_path.parts[:1], 'Masks', image_path.name)
        image = Image.open(args.data_root / image_path).convert('RGB')
        mask = Image.open(args.data_root / mask_path).convert('RGB')
        assert image.size == mask.size
        x, y, w, h = entry['crop_xywh']
        box = (x, y, x+w, y+h)
        assert w == h and 0 <= x < x+w <= image.width and 0 <= y < y+h <= image.height
        image = image.crop(box); mask = np.asarray(mask.crop(box))
        overlay = np.array(image, dtype=float)
        for channel, color, alpha in [(1, IRIS, .6), (2, PUPIL, 1.)]:
            selected = mask[:, :, channel] > 127
            overlay[selected] = (1-alpha) * overlay[selected] + alpha * np.array(color)
        overlay = Image.fromarray(overlay.clip(0,255).astype('uint8'))
        for row, tile in enumerate([image, overlay]):
            figure.paste(tile.resize((size,size), Image.Resampling.LANCZOS),
                         (padding + i*(size+gap), padding + row*(size+gap)))
        records.append({'example':name,'image':image_path.as_posix(),'mask':mask_path.as_posix(),
                        'crop_xywh':[x,y,w,h], 'image_sha256':hashlib.sha256((args.data_root/image_path).read_bytes()).hexdigest(),
                        'mask_sha256':hashlib.sha256((args.data_root/mask_path).read_bytes()).hexdigest()})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    figure.save(args.output)
    args.output.with_suffix('.json').write_text(json.dumps({'kind':'reference_masks_not_predictions','iris_rgb':IRIS,'pupil_rgb':PUPIL,'iris_alpha':.6,'pupil_alpha':1,'records':records},indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
