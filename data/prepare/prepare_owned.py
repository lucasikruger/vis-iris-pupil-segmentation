"""Rebuild consented/generated eye crops from their recorded original boxes."""
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.crop_metadata import load_boxes, relative_path, crop_from_record


def prepare_owned(source, args):
    names = ['white_eyes_train', 'white_eyes_test'] if source == 'white_eyes' else ['webcam']
    count = 0
    mismatches = []
    for name in names:
        records = load_boxes(args.annotations, name)
        rows = list(records.items())
        if args.limit:
            rows = rows[:args.limit]
        for stem, record in rows:
            dest = relative_path(args.output, record['image_path'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            if source == 'webcam':
                # Only archived eye crops are distributed; source coordinates remain metadata.
                shutil.copy2(relative_path(args.raw, record['image_path']), dest)
            else:
                image = cv2.imread(str(relative_path(args.raw, record['source_image'])))
                crop = crop_from_record(image, record)
                if not cv2.imwrite(str(dest), crop):
                    raise OSError(dest)
            reference = relative_path(args.annotations, record['image_path'])
            if hashlib.sha256(dest.read_bytes()).digest() != hashlib.sha256(reference.read_bytes()).digest():
                mismatches.append(record['image_path'])
            mdest = relative_path(args.output, record['mask_path'])
            mdest.parent.mkdir(parents=True, exist_ok=True)
            if source == 'white_eyes':
                cv2.imwrite(str(mdest), np.zeros_like(crop))
            elif record['has_mask']:
                shutil.copy2(relative_path(args.annotations, record['mask_path']), mdest)
            if mdest.exists():
                ref = relative_path(args.annotations, record['mask_path'])
                if hashlib.sha256(mdest.read_bytes()).digest() != hashlib.sha256(ref.read_bytes()).digest():
                    mismatches.append(record['mask_path'])
            count += 1
        metadata_dest = Path(args.output) / 'annotations' / f'{name}_boxes.json'
        metadata_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(args.annotations) / 'annotations' / metadata_dest.name, metadata_dest)
        # Keep labels and geometry available for evaluation. no source images are copied.
        dataset = Path(next(iter(records.values()))['image_path']).parts[0]
        for file in (Path(args.annotations) / dataset).iterdir():
            if file.is_file() and file.suffix in ['.csv', '.json']:
                shutil.copy2(file, Path(args.output) / dataset / file.name)
    report = {'source': source, 'reconstructed': count, 'sha256_mismatches': mismatches}
    (Path(args.output) / f'prepare_{source}_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit('Reconstructed crops differ from archived files')
