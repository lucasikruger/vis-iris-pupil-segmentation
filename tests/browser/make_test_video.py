"""Cámara falsa vertical con un rostro generado, sin capturas de personas reales."""
import argparse
from pathlib import Path
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/webcam.y4m'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-loop','1',
                    '-i',str(root/'demos/web/public/examples/white-eyes.webp'),
                    '-vf','crop=540:960:370:0,scale=360:640','-t','30','-r','15',
                    '-pix_fmt','yuv420p',str(args.output)],check=True)
