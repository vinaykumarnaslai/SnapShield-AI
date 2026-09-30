import argparse, time
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort

parser=argparse.ArgumentParser()
parser.add_argument('--model', default='models/yolov5n.onnx')
parser.add_argument('--iterations', type=int, default=30)
args=parser.parse_args()

model=Path(args.model)
if not model.exists():
    raise SystemExit('Model not found. Run: python scripts/download_model.py')

providers=ort.get_available_providers()
ordered=['QNNExecutionProvider','CPUExecutionProvider'] if 'QNNExecutionProvider' in providers else ['CPUExecutionProvider']
opts=ort.SessionOptions(); opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess=ort.InferenceSession(str(model), sess_options=opts, providers=ordered)
name=sess.get_inputs()[0].name
x=np.zeros((1,3,640,640),dtype=np.float32)
for _ in range(5): sess.run(None,{name:x})
start=time.perf_counter()
for _ in range(args.iterations): sess.run(None,{name:x})
elapsed=(time.perf_counter()-start)*1000/args.iterations
print('Provider order:', sess.get_providers())
print(f'Average inference request: {elapsed:.2f} ms over {args.iterations} iterations')
print('Note: provider order does not prove every model node executed on NPU; validate with QNN profiling on the target Snapdragon device.')
