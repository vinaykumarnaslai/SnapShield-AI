import base64
import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from app.detector import PrivacyDetector

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "demo" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
detector = PrivacyDetector(BASE_DIR / "models" / "yolov5n.onnx")


def jpeg_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise ValueError("Could not encode image")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, **detector.runtime_info()})


@app.post("/api/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    raw = file.read()
    if not raw:
        return jsonify({"error": "The uploaded file is empty"}), 400

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Unsupported image format"}), 400

    strict = request.form.get("strict_npu") == "1"
    policy = request.form.get("policy", "people")
    start = time.perf_counter()
    result = detector.analyze(image, policy=policy, strict_npu=strict)
    elapsed = (time.perf_counter() - start) * 1000

    return jsonify({
        "original": jpeg_data_url(image),
        "redacted": jpeg_data_url(result["image"]),
        "detections": result["detections"],
        "faces": result["faces"],
        "latency_ms": round(elapsed, 1),
        "policy": policy,
        **detector.runtime_info(),
    })
