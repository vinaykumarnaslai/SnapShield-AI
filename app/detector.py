from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

POLICIES = {
    "people": {"person"},
    "people_devices": {"person", "cell phone", "laptop"},
    "privacy_max": {"person", "cell phone", "laptop", "book", "keyboard"},
}


class PrivacyDetector:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.input_size = 640
        self.session = None
        self.active_provider = "CPUExecutionProvider"
        self._load_session()
        cascade_dir = Path(cv2.data.haarcascades)

        self.face = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_frontalface_default.xml")
        )

        self.face_alt = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_frontalface_alt2.xml")
        )

        self.profile_face = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_profileface.xml")
        )

    def _load_session(self, strict_npu: bool = False):
        providers = ort.get_available_providers()
        preferred = ["QNNExecutionProvider", "CPUExecutionProvider"] if "QNNExecutionProvider" in providers else ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if strict_npu and "QNNExecutionProvider" in providers:
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if self.model_path.exists():
            self.session = ort.InferenceSession(str(self.model_path), sess_options=options, providers=preferred)
            session_providers = self.session.get_providers()
            self.active_provider = session_providers[0] if session_providers else preferred[0]

    def runtime_info(self) -> dict[str, Any]:
        available = ort.get_available_providers()
        qnn = "QNNExecutionProvider" in available
        return {
            "model": self.model_path.name,
            "model_available": self.model_path.exists(),
            "available_providers": available,
            "execution_provider": self.active_provider,
            "target_backend": "HTP / NPU" if qnn else "CPU fallback",
            "snapdragon_ready": qnn,
            "optimization_note": "QNN is selected first when available; CPU remains a development fallback."
        }

    def _letterbox(self, image: np.ndarray):
        h, w = image.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - nw) // 2
        pad_y = (self.input_size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y

    def _postprocess(self, output: np.ndarray, original_shape, scale, pad_x, pad_y, conf=0.35):
        if output.ndim == 3:
            output = output[0]
        if output.shape[0] < output.shape[1] and output.shape[0] <= 100:
            output = output.T
        h, w = original_shape[:2]
        detections = []
        boxes, scores, class_ids = [], [], []
        for row in output:
            objectness = float(row[4])
            if objectness <= 0:
                continue
            class_scores = row[5:]
            cls = int(np.argmax(class_scores))
            score = objectness * float(class_scores[cls])
            if score < conf:
                continue
            x, y, bw, bh = row[:4]
            x1 = int(max(0, (x - bw / 2 - pad_x) / scale))
            y1 = int(max(0, (y - bh / 2 - pad_y) / scale))
            x2 = int(min(w - 1, (x + bw / 2 - pad_x) / scale))
            y2 = int(min(h - 1, (y + bh / 2 - pad_y) / scale))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(cls)
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf, 0.45)
        if len(indices) == 0:
            return []
        indices = np.array(indices).reshape(-1)
        for idx in indices:
            cls_id = class_ids[int(idx)]
            detections.append({
                "class": COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else str(cls_id),
                "confidence": round(float(scores[int(idx)]), 3),
                "box": boxes[int(idx)],
            })
        return detections

    def _redact_box(self, image, box, label):
        x, y, w, h = box
        x = max(0, x); y = max(0, y)
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return
        k = max(15, (min(roi.shape[:2]) // 9) * 2 + 1)
        k = min(k, 99)
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        image[y:y+h, x:x+w] = blurred

    def _blur_region(self, image, x0, y0, x1, y1, label="face"):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(image.shape[1], int(x1)), min(image.shape[0], int(y1))
        if x1 <= x0 or y1 <= y0:
            return False
        roi = image[y0:y1, x0:x1]
        if roi.size == 0:
            return False
        k = max(21, (min(roi.shape[:2]) // 7) * 2 + 1)
        k = min(k, 101)
        image[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
        return True

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        ax2, ay2, bx2, by2 = ax+aw, ay+ah, bx+bw, by+bh
        ix0, iy0, ix1, iy1 = max(ax,bx), max(ay,by), min(ax2,bx2), min(ay2,by2)
        iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
        inter = iw*ih
        union = aw*ah + bw*bh - inter
        return inter/union if union else 0.0

    def _redact_faces(self, image, person_boxes=None):
        """
        Detect and redact one face per detected person.

        The YOLO person detections are used as hard search regions.
        Multiple Haar cascades can propose candidates, but only the
        best candidate for each person is retained. No face position
        is guessed when a detector fails.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        person_boxes = person_boxes or []

        def clamp_box(box):
            x, y, w, h = map(int, box)
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            return (x, y, w, h)

        def valid_shape(w, h):
            if w < 24 or h < 24:
                return False

            ratio = w / float(h)

            # Haar face boxes should be roughly square/portrait.
            if ratio < 0.65 or ratio > 1.45:
                return False

            return True

        def candidate_score(box, person):
            """
            Lower is better.

            Prefer:
            - detections near the expected head area
            - reasonable face size
            - detections that are not too close to the edges
            """
            x, y, w, h = box
            px, py, pw, ph = person

            cx = x + w / 2.0
            cy = y + h / 2.0

            expected_cx = px + pw / 2.0
            expected_cy = py + ph * 0.22

            dx = abs(cx - expected_cx) / max(pw, 1)
            dy = abs(cy - expected_cy) / max(ph, 1)

            distance = dx * 0.65 + dy * 1.35

            # Prefer a face that has a reasonable size relative to
            # the detected person.
            size_ratio = w / max(pw, 1)

            size_penalty = 0.0

            if size_ratio < 0.08:
                size_penalty += 0.35

            if size_ratio > 0.38:
                size_penalty += 0.35

            return distance + size_penalty

        def add_candidates(
            crop,
            offset_x,
            offset_y,
            person,
            candidates
        ):
            """
            Run all Haar detectors on one person's head region.
            """
            detectors = [
                self.face,
                self.face_alt,
                self.profile_face,
            ]

            for detector in detectors:
                if detector.empty():
                    continue

                found = detector.detectMultiScale(
                    crop,
                    scaleFactor=1.08,
                    minNeighbors=6,
                    minSize=(24, 24),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )

                for x, y, w, h in found:
                    if not valid_shape(w, h):
                        continue

                    box = clamp_box(
                        (
                            offset_x + x,
                            offset_y + y,
                            w,
                            h,
                        )
                    )

                    if not valid_shape(box[2], box[3]):
                        continue

                    candidates.append(
                        (
                            candidate_score(box, person),
                            box,
                        )
                    )

            # Profile detector on horizontally flipped crop.
            if not self.profile_face.empty():
                flipped = cv2.flip(crop, 1)

                found = self.profile_face.detectMultiScale(
                    flipped,
                    scaleFactor=1.08,
                    minNeighbors=6,
                    minSize=(24, 24),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )

                crop_width = crop.shape[1]

                for x, y, w, h in found:
                    if not valid_shape(w, h):
                        continue

                    original_x = crop_width - x - w

                    box = clamp_box(
                        (
                            offset_x + original_x,
                            offset_y + y,
                            w,
                            h,
                        )
                    )

                    if not valid_shape(box[2], box[3]):
                        continue

                    candidates.append(
                        (
                            candidate_score(box, person),
                            box,
                        )
                    )

        # ---------------------------------------------------------
        # Primary path: one selected face for each YOLO person.
        # ---------------------------------------------------------
        if person_boxes:
            selected_faces = []

            for person in person_boxes:
                px, py, pw, ph = map(int, person)

                if pw <= 0 or ph <= 0:
                    continue

                # Only search the head/upper torso region.
                # This avoids laptops, tables and lower-body details.
                rx0 = max(0, px)
                ry0 = max(0, py)

                rx1 = min(
                    image.shape[1],
                    px + pw
                )

                ry1 = min(
                    image.shape[0],
                    py + int(ph * 0.55)
                )

                if rx1 <= rx0 or ry1 <= ry0:
                    continue

                crop = gray[ry0:ry1, rx0:rx1]

                if crop.size == 0:
                    continue

                candidates = []

                add_candidates(
                    crop,
                    rx0,
                    ry0,
                    (px, py, pw, ph),
                    candidates,
                )

                if not candidates:
                    continue

                # Remove duplicate candidates for this person using IoU.
                candidates.sort(key=lambda item: item[0])

                best = None

                for score, box in candidates:
                    if best is None:
                        best = (score, box)
                        continue

                    # If candidate overlaps the current best, keep the
                    # candidate with the better head-position score.
                    if self._iou(box, best[1]) > 0.25:
                        continue

                    # A different candidate is only useful if it is
                    # materially better.
                    if score < best[0] - 0.15:
                        best = (score, box)

                if best is not None:
                    selected_faces.append(best[1])

            # -----------------------------------------------------
            # Global deduplication.
            # -----------------------------------------------------
            unique_faces = []

            for face in selected_faces:
                if not any(
                    self._iou(face, old) > 0.35
                    for old in unique_faces
                ):
                    unique_faces.append(face)

            # -----------------------------------------------------
            # Redact the selected faces.
            # -----------------------------------------------------
            for x, y, w, h in unique_faces:
                pad_x = int(w * 0.12)
                pad_y = int(h * 0.12)

                self._blur_region(
                    image,
                    x - pad_x,
                    y - pad_y,
                    x + w + pad_x,
                    y + h + pad_y,
                    "face",
                )

            return len(unique_faces)

        # ---------------------------------------------------------
        # Fallback: no person detections, so use global face search.
        # ---------------------------------------------------------
        candidates = []

        for detector in [
            self.face,
            self.face_alt,
            self.profile_face,
        ]:
            if detector.empty():
                continue

            found = detector.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=7,
                minSize=(28, 28),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            for x, y, w, h in found:
                if valid_shape(w, h):
                    candidates.append(
                        (
                            int(x),
                            int(y),
                            int(w),
                            int(h),
                        )
                    )

        unique_faces = []

        for face in candidates:
            if not any(
                self._iou(face, old) > 0.35
                for old in unique_faces
            ):
                unique_faces.append(face)

        for x, y, w, h in unique_faces:
            pad_x = int(w * 0.12)
            pad_y = int(h * 0.12)

            self._blur_region(
                image,
                x - pad_x,
                y - pad_y,
                x + w + pad_x,
                y + h + pad_y,
                "face",
            )

        return len(unique_faces)

    def analyze(self, image: np.ndarray, policy="people", strict_npu=False):
        if not self.model_path.exists() or self.session is None:
            raise RuntimeError("Model missing. Run: python scripts/download_model.py")
        self._load_session(strict_npu=strict_npu)
        canvas, scale, px, py = self._letterbox(image)
        inp = canvas[:, :, ::-1].astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None]
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: inp})[0]
        detections = self._postprocess(output, image.shape, scale, px, py)
        selected = POLICIES.get(policy, POLICIES["people"])
        redacted = image.copy()
        # Faces are the primary privacy target for the People policy.
        # Person detections are used as contextual evidence/counting, but we do
        # not blur the entire person bounding box because that over-redacts the
        # image. Non-person sensitive classes (for example phones or laptops)
        # are still redacted using their YOLO boxes.
        kept = [d for d in detections if d["class"] in selected]
        object_redactions = []
        for d in kept:
            if d["class"] != "person":
                self._redact_box(redacted, d["box"], d["class"])
                object_redactions.append(d)

        # Run face detection on the original image. Person boxes provide a
        # targeted fallback for profile/occluded faces that Haar may miss.
        face_redacted = image.copy()
        person_boxes = [d["box"] for d in kept if d["class"] == "person"]
        faces = self._redact_faces(face_redacted, person_boxes=person_boxes)

        if policy == "people":
            redacted = face_redacted
        else:
            # Add face redaction to the other selected-object redactions.
            redacted = face_redacted
            for d in object_redactions:
                # Re-apply non-person redactions on top of the face-redacted copy.
                self._redact_box(redacted, d["box"], d["class"])

        total_redactions = faces + len(object_redactions)
        return {
            "image": redacted,
            "detections": kept,
            "object_redactions": object_redactions,
            "faces": faces,
            "redaction_count": total_redactions,
        }
