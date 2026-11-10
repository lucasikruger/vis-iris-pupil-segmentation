"""Valida el pool reconstruido contra los hashes del pool original."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='data/mobius_format');a=p.parse_args()
manifest=json.loads(Path('data/prepared_sha256.json').read_text());missing=[];different=[]
for rel,expected in manifest.items():
 path=Path(a.root)/rel
 if not path.is_file():missing.append(rel)
 elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:different.append(rel)
print(json.dumps({'expected':len(manifest),'missing':len(missing),'different':different},indent=2))
if missing or different:raise SystemExit(1)
