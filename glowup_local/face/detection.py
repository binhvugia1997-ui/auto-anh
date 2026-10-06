"""Offline face detection with OpenCV's bundled Haar cascades.

Haar cascades do not provide a dense face mesh. We combine their face/eye detections with
conservative, normalized estimates for the nose, mouth and face contour. The data model is
intentionally replaceable by a denser local landmark provider in a future release.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _point(value: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    return (
        float(np.clip(value[0] / max(1, width), 0.0, 1.0)),
        float(np.clip(value[1] / max(1, height), 0.0, 1.0)),
    )


@dataclass(frozen=True, slots=True)
class FaceRegion:
    """Face rectangle and feature centers in image-relative (0..1) coordinates."""

    bbox: tuple[float, float, float, float]
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    mouth: tuple[float, float]
    confidence: float = 0.5
    landmarks: tuple[tuple[float, float], ...] = ()

    def pixel_bbox(self, width: int, height: int) -> tuple[int, int, int, int]:
        x, y, w, h = self.bbox
        x0 = round(x * width)
        y0 = round(y * height)
        x1 = round((x + w) * width)
        y1 = round((y + h) * height)
        x0, y0 = max(0, min(width - 1, x0)), max(0, min(height - 1, y0))
        x1, y1 = max(x0 + 1, min(width, x1)), max(y0 + 1, min(height, y1))
        return x0, y0, x1 - x0, y1 - y0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaceRegion:
        return cls(
            bbox=tuple(float(v) for v in data["bbox"]),
            left_eye=tuple(float(v) for v in data["left_eye"]),
            right_eye=tuple(float(v) for v in data["right_eye"]),
            nose=tuple(float(v) for v in data["nose"]),
            mouth=tuple(float(v) for v in data["mouth"]),
            confidence=float(data.get("confidence", 0.5)),
            landmarks=tuple(
                tuple(float(v) for v in point) for point in data.get("landmarks", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class FaceAnalysis:
    faces: tuple[FaceRegion, ...] = ()
    method: str = "opencv-haar"

    def to_dict(self) -> dict[str, Any]:
        return {"faces": [face.to_dict() for face in self.faces], "method": self.method}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FaceAnalysis:
        if not data:
            return cls()
        return cls(
            tuple(FaceRegion.from_dict(f) for f in data.get("faces", ())),
            str(data.get("method", "opencv-haar")),
        )


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _choose_eyes(
    gray: np.ndarray, x: int, y: int, w: int, h: int, eye_cascade: cv2.CascadeClassifier
) -> tuple[tuple[int, int], tuple[int, int]]:
    roi_x0, roi_x1 = x + int(0.08 * w), x + int(0.92 * w)
    roi_y0, roi_y1 = y + int(0.16 * h), y + int(0.57 * h)
    roi = gray[
        max(0, roi_y0) : min(gray.shape[0], roi_y1),
        max(0, roi_x0) : min(gray.shape[1], roi_x1),
    ]
    candidates: list[tuple[int, int, int, int]] = []
    if roi.size:
        try:
            for ex, ey, ew, eh in eye_cascade.detectMultiScale(
                roi,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(
                    max(8, int(w * 0.045)),
                    max(7, int(h * 0.035)),
                ),
            ):
                candidates.append(
                    (int(ex + roi_x0), int(ey + roi_y0), int(ew), int(eh))
                )
        except cv2.error as exc:
            logger.debug("OpenCV eye detection failed for one face region: %s", exc)
    mid = x + w * 0.5
    target_y = y + h * 0.40
    picked: list[tuple[int, int] | None] = [None, None]
    for side, (lo, hi) in enumerate(
        ((x + 0.12 * w, mid - 0.02 * w), (mid + 0.02 * w, x + 0.88 * w))
    ):
        valid = [
            box
            for box in candidates
            if lo <= box[0] + box[2] / 2 <= hi
            and y + 0.18 * h <= box[1] + box[3] / 2 <= y + 0.56 * h
        ]
        if valid:
            # Prefer a sizeable detection near the expected eye line, not eyebrows/noise.
            best = max(
                valid,
                key=lambda b: (
                    b[2] * b[3] / (1 + abs((b[1] + b[3] / 2) - target_y) / max(1, h))
                ),
            )
            picked[side] = (best[0] + best[2] // 2, best[1] + best[3] // 2)
    estimated = [
        (int(x + 0.32 * w), int(y + 0.40 * h)),
        (int(x + 0.68 * w), int(y + 0.40 * h)),
    ]
    if picked[0] is None and picked[1] is not None:
        picked[0] = (int(2 * mid - picked[1][0]), picked[1][1])
    if picked[1] is None and picked[0] is not None:
        picked[1] = (int(2 * mid - picked[0][0]), picked[0][1])
    return picked[0] or estimated[0], picked[1] or estimated[1]


_FACE_OVAL = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)
_EYE_A = (33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155)
_EYE_B = (362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382)
_LIP_OUTER = (
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
    375,
    321,
    405,
    314,
    17,
    84,
    181,
    91,
    146,
)


def _detect_mediapipe(rgb: np.ndarray) -> FaceAnalysis | None:
    """Use the wheel-bundled Face Mesh model when the optional runtime is installed."""
    try:
        import mediapipe as mp  # optional; no import or download is required by the base app

        solutions = getattr(mp, "solutions", None)
        face_mesh_module = getattr(solutions, "face_mesh", None) if solutions else None
        if face_mesh_module is None:
            return None
        with face_mesh_module.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.45,
        ) as mesh:
            result = mesh.process(np.ascontiguousarray(rgb))
        faces: list[FaceRegion] = []
        for face_landmarks in result.multi_face_landmarks or ():
            points = tuple(
                (float(np.clip(point.x, 0, 1)), float(np.clip(point.y, 0, 1)))
                for point in face_landmarks.landmark
            )
            if len(points) < 468:
                continue
            oval = [points[index] for index in _FACE_OVAL]
            min_x = min(p[0] for p in oval)
            max_x = max(p[0] for p in oval)
            min_y = min(p[1] for p in oval)
            max_y = max(p[1] for p in oval)
            if max_x - min_x < 0.025 or max_y - min_y < 0.025:
                continue

            def center(
                indices: tuple[int, ...],
                landmarks: tuple[tuple[float, float], ...] = points,
            ) -> tuple[float, float]:
                return (
                    float(np.mean([landmarks[i][0] for i in indices])),
                    float(np.mean([landmarks[i][1] for i in indices])),
                )

            left_eye = center(_EYE_A)
            right_eye = center(_EYE_B)
            nose = points[1]
            upper_lip, lower_lip = points[13], points[14]
            mouth = (
                (upper_lip[0] + lower_lip[0]) * 0.5,
                (upper_lip[1] + lower_lip[1]) * 0.5,
            )
            faces.append(
                FaceRegion(
                    bbox=(min_x, min_y, max_x - min_x, max_y - min_y),
                    left_eye=left_eye,
                    right_eye=right_eye,
                    nose=nose,
                    mouth=mouth,
                    confidence=0.92,
                    landmarks=points,
                )
            )
        return FaceAnalysis(tuple(faces), "mediapipe-face-mesh")
    except Exception as exc:  # noqa: BLE001 - optional/incompatible model must never block imports
        logger.debug(
            "Optional MediaPipe Face Mesh unavailable; using Haar fallback: %s", exc
        )
        return None


def detect_faces(image: np.ndarray, max_dimension: int = 1600) -> FaceAnalysis:
    """Find frontal/profile faces and approximate facial landmarks entirely offline.

    Detection is performed on a bounded-size grayscale copy, so a 48 MP photo does not
    allocate a full-resolution detection pyramid. Empty/corrupt inputs safely return no faces.
    """
    if image is None or image.size == 0 or image.ndim not in (2, 3):
        return FaceAnalysis()
    try:
        height, width = image.shape[:2]
        if width < 20 or height < 20:
            return FaceAnalysis()
        if image.ndim == 2:
            gray_source = image
        elif image.shape[2] == 4:
            gray_source = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray_source = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
        scale = min(1.0, max_dimension / float(max(width, height)))
        if scale < 1:
            work_width = max(1, round(width * scale))
            work_height = max(1, round(height * scale))
            gray = cv2.resize(
                gray_source, (work_width, work_height), interpolation=cv2.INTER_AREA
            )
        else:
            work_width, work_height = width, height
            gray = gray_source
        if image.ndim == 3:
            rgb_work = image[..., :3]
            if scale < 1:
                rgb_work = cv2.resize(
                    rgb_work, (work_width, work_height), interpolation=cv2.INTER_AREA
                )
            mesh_analysis = _detect_mediapipe(rgb_work)
            if mesh_analysis is not None and mesh_analysis.faces:
                return mesh_analysis
        gray = cv2.equalizeHist(gray)
        cascade_root = cv2.data.haarcascades
        face_cascade = cv2.CascadeClassifier(
            cascade_root + "haarcascade_frontalface_default.xml"
        )
        profile_cascade = cv2.CascadeClassifier(
            cascade_root + "haarcascade_profileface.xml"
        )
        eye_cascade = cv2.CascadeClassifier(
            cascade_root + "haarcascade_eye_tree_eyeglasses.xml"
        )
        if face_cascade.empty():
            logger.error("OpenCV frontal-face cascade is unavailable")
            return FaceAnalysis((), "opencv-haar-unavailable")
        min_size = max(36, int(min(gray.shape[:2]) * 0.055))
        boxes: list[tuple[int, int, int, int]] = [
            tuple(map(int, b))
            for b in face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_size, min_size),
            )
        ]
        if not profile_cascade.empty():
            for prof in profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_size, min_size),
            ):
                candidate = tuple(map(int, prof))
                if all(_iou(candidate, current) < 0.42 for current in boxes):
                    boxes.append(candidate)
            flipped = cv2.flip(gray, 1)
            for px, py, pw, ph in profile_cascade.detectMultiScale(
                flipped,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_size, min_size),
            ):
                candidate = (
                    gray.shape[1] - int(px) - int(pw),
                    int(py),
                    int(pw),
                    int(ph),
                )
                if all(_iou(candidate, current) < 0.42 for current in boxes):
                    boxes.append(candidate)
        # Prefer larger detections and discard nested duplicate cascade hits.
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        unique: list[tuple[int, int, int, int]] = []
        for box in boxes:
            if all(_iou(box, other) < 0.48 for other in unique):
                unique.append(box)
        work_h, work_w = gray.shape[:2]
        found: list[FaceRegion] = []
        for x, y, w, h in unique:
            x, y = max(0, x), max(0, y)
            w, h = min(w, work_w - x), min(h, work_h - y)
            if w < 16 or h < 16:
                continue
            eyes = _choose_eyes(gray, x, y, w, h, eye_cascade)
            left_eye = _point(eyes[0], work_w, work_h)
            right_eye = _point(eyes[1], work_w, work_h)
            found.append(
                FaceRegion(
                    bbox=(x / work_w, y / work_h, w / work_w, h / work_h),
                    left_eye=left_eye,
                    right_eye=right_eye,
                    nose=_point((int(x + 0.5 * w), int(y + 0.56 * h)), work_w, work_h),
                    mouth=_point((int(x + 0.5 * w), int(y + 0.73 * h)), work_w, work_h),
                    confidence=0.55 if w < min_size * 1.4 else 0.68,
                )
            )
        return FaceAnalysis(tuple(found))
    except (cv2.error, ValueError, IndexError) as exc:
        logger.warning("Face detection failed gracefully: %s", exc)
        return FaceAnalysis()
