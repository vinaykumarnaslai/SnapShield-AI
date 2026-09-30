

### Redaction behavior

The default **People + faces** policy uses YOLO person detections for context but does not blur the full person bounding box. It uses OpenCV face detection to redact the face region, avoiding unnecessary over-redaction. Device/object policies additionally redact selected non-person objects.
