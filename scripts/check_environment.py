import platform
try:
    import onnxruntime as ort
except Exception as exc:
    print('onnxruntime import failed:', exc)
    raise SystemExit(1)

print('OS:', platform.platform())
print('Machine:', platform.machine())
print('ONNX Runtime:', ort.__version__)
print('Available providers:', ort.get_available_providers())
print('QNN available:', 'QNNExecutionProvider' in ort.get_available_providers())
