"""Una época pequeña: negativas propias o FEI obtenido con su licencia. nunca webcam."""
import argparse,subprocess,sys
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',choices=['white_eyes_bg','fei_manual'],default='white_eyes_bg')
p.add_argument('--data-root',default='dataset-package');a=p.parse_args()
out=Path('runs')/('smoke-'+a.source);out.mkdir(parents=True,exist_ok=True)
x=pd.read_csv('data/partitions.csv');x=x[x.dataset==a.source]
if a.source=='white_eyes_bg':
 x=x.head(32).copy();x['dataset']='smoke_white_eyes'
 x['subject_id']=['smoke/'+str(i//2) for i in range(len(x))]
else:
 x=pd.concat([x[x.split=='train'].head(16),x[x.split=='val'].head(8)])
x.to_csv(out/'samples.csv',index=False)
subprocess.run([sys.executable,'scripts/train.py','--csv',str(out/'samples.csv'),'--data-root',a.data_root,
                '--epochs','1','--output',str(out)],check=True)
