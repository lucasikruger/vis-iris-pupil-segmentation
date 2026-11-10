"""Exporta el checkpoint elegido a ONNX (logits, RGB normalizado, 96×96)."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from src.models import build_model
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--weights',default='weights/model.pt')
p.add_argument('--output',default='runs/model.onnx');a=p.parse_args()
torch.set_num_threads(4)
m=build_model();m.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True)['model_state']);m.eval()
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
torch.onnx.export(m,torch.randn(1,3,96,96),a.output,input_names=['input'],output_names=['logits'],opset_version=17,dynamo=False)
print(a.output)
