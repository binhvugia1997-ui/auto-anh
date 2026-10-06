"""Soft facial region masks derived from normalized face/eye landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..face.detection import FaceRegion

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
_EYE_A = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
_EYE_B = (
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    398,
    384,
    385,
    386,
    387,
    388,
    466,
)
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
_LIP_INNER = (
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
)


@dataclass(slots=True)
class FaceMasks:
    rect: tuple[int, int, int, int]  # x0, y0, x1, y1
    skin: np.ndarray
    eyes: np.ndarray
    under_eyes: np.ndarray
    lips: np.ndarray
    teeth: np.ndarray


def _ellipse(
    mask: np.ndarray, center: tuple[float, float], axes: tuple[float, float], value: int
) -> None:
    cx, cy = round(center[0]), round(center[1])
    ax, ay = max(1, round(axes[0])), max(1, round(axes[1]))
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, value, -1, cv2.LINE_AA)


def _soften(mask: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(0.7, float(sigma))
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _landmark_polygon(
    mask: np.ndarray,
    landmarks: tuple[tuple[float, float], ...],
    indices: tuple[int, ...],
    image_width: int,
    image_height: int,
    x0: int,
    y0: int,
    value: int = 255,
) -> bool:
    if not landmarks or any(index >= len(landmarks) for index in indices):
        return False
    points = np.array(
        [
            (
                landmarks[index][0] * image_width - x0,
                landmarks[index][1] * image_height - y0,
            )
            for index in indices
        ],
        dtype=np.float32,
    )
    points[:, 0] = np.clip(points[:, 0], 0, mask.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, mask.shape[0] - 1)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], value, cv2.LINE_AA)
    return True


def masks_for_face(shape: tuple[int, ...], face: FaceRegion) -> FaceMasks:
    """Return small face-local masks; no full-canvas temporary mask is allocated."""
    height, width = shape[:2]
    fx, fy, fw, fh = face.pixel_bbox(width, height)
    pad_x, pad_y = max(3, int(fw * 0.07)), max(3, int(fh * 0.07))
    x0, y0 = max(0, fx - pad_x), max(0, fy - pad_y)
    x1, y1 = min(width, fx + fw + pad_x), min(height, fy + fh + pad_y)
    rw, rh = max(1, x1 - x0), max(1, y1 - y0)
    skin = np.zeros((rh, rw), np.uint8)
    eyes = np.zeros_like(skin)
    under = np.zeros_like(skin)
    lips = np.zeros_like(skin)
    teeth = np.zeros_like(skin)

    def local(norm_point: tuple[float, float]) -> tuple[float, float]:
        return norm_point[0] * width - x0, norm_point[1] * height - y0

    # Face contour is approximated from the detector rectangle. It deliberately excludes
    # the hairline and outer ears, where broad skin smoothing tends to create artifacts.
    landmarks = face.landmarks
    if not _landmark_polygon(skin, landmarks, _FACE_OVAL, width, height, x0, y0):
        polygon = np.array(
            [
                (fx + 0.50 * fw, fy + 0.10 * fh),
                (fx + 0.76 * fw, fy + 0.15 * fh),
                (fx + 0.91 * fw, fy + 0.34 * fh),
                (fx + 0.96 * fw, fy + 0.56 * fh),
                (fx + 0.85 * fw, fy + 0.77 * fh),
                (fx + 0.67 * fw, fy + 0.94 * fh),
                (fx + 0.50 * fw, fy + 0.99 * fh),
                (fx + 0.33 * fw, fy + 0.94 * fh),
                (fx + 0.15 * fw, fy + 0.77 * fh),
                (fx + 0.04 * fw, fy + 0.56 * fh),
                (fx + 0.09 * fw, fy + 0.34 * fh),
                (fx + 0.24 * fw, fy + 0.15 * fh),
            ],
            dtype=np.float32,
        )
        polygon[:, 0] -= x0
        polygon[:, 1] -= y0
        cv2.fillPoly(skin, [np.round(polygon).astype(np.int32)], 255, cv2.LINE_AA)

    eye_points = (face.left_eye, face.right_eye)
    eyes_local = [local(p) for p in eye_points]
    if landmarks:
        _landmark_polygon(eyes, landmarks, _EYE_A, width, height, x0, y0)
        _landmark_polygon(eyes, landmarks, _EYE_B, width, height, x0, y0)
    for point in eyes_local:
        if not landmarks:
            _ellipse(eyes, point, (0.105 * fw, 0.052 * fh), 255)
        # Remove lids, lashes and eyebrows from the skin blur mask.
        _ellipse(skin, point, (0.135 * fw, 0.095 * fh), 0)
        _ellipse(skin, (point[0], point[1] - 0.105 * fh), (0.15 * fw, 0.042 * fh), 0)
        _ellipse(under, (point[0], point[1] + 0.075 * fh), (0.13 * fw, 0.055 * fh), 255)

    mouth = local(face.mouth)
    if not _landmark_polygon(lips, landmarks, _LIP_OUTER, width, height, x0, y0):
        _ellipse(lips, mouth, (0.225 * fw, 0.082 * fh), 255)
    if not _landmark_polygon(teeth, landmarks, _LIP_INNER, width, height, x0, y0):
        _ellipse(
            teeth, (mouth[0], mouth[1] + 0.018 * fh), (0.105 * fw, 0.033 * fh), 255
        )
    _ellipse(skin, mouth, (0.25 * fw, 0.105 * fh), 0)
    nose = local(face.nose)
    _ellipse(skin, (nose[0], nose[1] + 0.055 * fh), (0.09 * fw, 0.035 * fh), 0)

    softness = max(1.0, fw * 0.018)
    return FaceMasks(
        (x0, y0, x1, y1),
        _soften(skin, softness),
        _soften(eyes, max(0.7, fw * 0.009)),
        _soften(under, max(0.7, fw * 0.012)),
        _soften(lips, max(0.7, fw * 0.01)),
        _soften(teeth, max(0.6, fw * 0.007)),
    )
