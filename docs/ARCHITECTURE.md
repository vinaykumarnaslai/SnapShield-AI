# SnapShield AI - Architecture

```text
                    +---------------------------+
                    |     Browser UI (Flask)    |
                    | upload -> analyze -> save |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |   PrivacyDetector         |
                    | - YOLOv5n ONNX            |
                    | - face detection          |
                    +-------------+-------------+
                                  |
                    QNN available?                
                     /          \
                   yes           no
                    |             |
                    v             v
            +-------------+  +------------+
            | QNN / HTP   |  | CPU EP     |
            | Snapdragon  |  | dev/fallback|
            +-------------+  +------------+
                    |
                    v
            redacted image +
            runtime telemetry
```

## Snapdragon optimization

- ONNX is the portable model interchange format.
- ONNX Runtime is configured to select `QNNExecutionProvider` first when it is available.
- QNN HTP is the NPU-oriented backend.
- CPU fallback keeps local development possible on non-Snapdragon systems.
- The Snapdragon deployment target is Windows 11 on ARM64 with `onnxruntime-qnn`.
- The app exposes the selected execution provider and inference latency in the UI so the deployment behavior is inspectable.

## Privacy flow

1. Image remains on the local machine.
2. YOLOv5n identifies selected COCO object classes.
3. OpenCV face detection adds face coverage.
4. Selected regions receive irreversible Gaussian blur.
5. The user can download the redacted copy.

## Evidence

Qualcomm AI Hub publishes a YOLOv5 job targeting Snapdragon X Elite CRD on Windows 11 using the NPU, with an estimated inference time of 14.8 ms and 47 MB peak memory for that profile. These are AI Hub reference measurements, not measurements claimed for this prototype.
