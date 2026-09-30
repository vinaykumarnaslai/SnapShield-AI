# Submission Notes

## Project
SnapShield AI - On-Device Privacy Redaction for Snapdragon PCs

## What is built
A local Flask application that accepts images, detects privacy-sensitive regions, applies local redaction, and reports the runtime provider and latency.

## Snapdragon relevance
The runtime prefers ONNX Runtime's Qualcomm QNN Execution Provider when installed. QNN supports Windows devices with Snapdragon SoCs and can use the HTP backend to offload inference to the NPU. The same application retains CPU fallback for development.

## Qualcomm AI Hub reference
Qualcomm AI Hub has a published YOLOv5 profile targeting Snapdragon X Elite CRD / Windows 11. Its listed profile reports 14.8 ms estimated inference time, 47 MB estimated peak memory, and NPU compute for that job.

## Important claim boundary
The 14.8 ms figure is a Qualcomm AI Hub reference result, not a measured result produced by this prototype. Replace it with your own benchmark numbers after running the project on a Snapdragon PC, if available.
