from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "yolov5n.onnx"
URL = "https://github.com/yakhyo/yolov5-onnx-inference/releases/download/v0.0.1/yolov5n.onnx"
OUT.parent.mkdir(exist_ok=True, parents=True)
if OUT.exists():
    print(f"Already present: {OUT}")
else:
    print("Downloading YOLOv5n ONNX (~8 MB)...")
    urlretrieve(URL, OUT)
    print(f"Saved: {OUT}")
