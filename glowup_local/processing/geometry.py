"""Local, smoothly feathered face/eye deformation using inverse-map remapping."""

from __future__ import annotations

import cv2
import numpy as np

from ..face.detection import FaceAnalysis, FaceRegion
from ..models.edit_state import EditState


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _blend(warped: np.ndarray, source: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weight = np.ascontiguousarray(np.clip(weight, 0, 1), dtype=np.float32)
    return cv2.blendLinear(warped, source, weight, 1.0 - weight)


def _local_rect(face: FaceRegion, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = face.pixel_bbox(width, height)
    padx, pady = max(4, int(w * 0.20)), max(4, int(h * 0.15))
    return (
        max(0, x - padx),
        max(0, y - pady),
        min(width, x + w + padx),
        min(height, y + h + pady),
    )


def _eye_warp(
    image: np.ndarray, face: FaceRegion, rect: tuple[int, int, int, int], strength: int
) -> None:
    """Enlarge each eye in a small eye-local ROI rather than allocating face-sized maps."""
    if strength <= 0:
        return
    x0, y0, x1, y1 = rect
    height, width = image.shape[:2]
    _, _, fw, fh = face.pixel_bbox(width, height)
    for point in (face.left_eye, face.right_eye):
        cx, cy = point[0] * width, point[1] * height
        rx, ry = max(2.0, 0.125 * fw), max(2.0, 0.072 * fh)
        ex0, ex1 = max(x0, int(cx - 2.4 * rx)), min(x1, int(cx + 2.4 * rx) + 1)
        ey0, ey1 = max(y0, int(cy - 2.4 * ry)), min(y1, int(cy + 2.4 * ry) + 1)
        if ex1 <= ex0 or ey1 <= ey0:
            continue
        gx = np.arange(ex0, ex1, dtype=np.float32)[None, :]
        gy = np.arange(ey0, ey1, dtype=np.float32)[:, None]
        dx, dy = (gx - cx) / rx, (gy - cy) / ry
        radius2 = dx * dx + dy * dy
        influence = np.exp(-1.15 * radius2).astype(np.float32)
        zoom = 1.0 + 0.13 * (strength / 100.0) * influence
        map_x = (
            np.broadcast_to((gx - cx) / zoom + cx - x0, (ey1 - ey0, ex1 - ex0))
            .copy()
            .astype(np.float32)
        )
        map_y = (
            np.broadcast_to((gy - cy) / zoom + cy - y0, (ey1 - ey0, ex1 - ex0))
            .copy()
            .astype(np.float32)
        )
        source_patch = image[y0:y1, x0:x1]
        warped = cv2.remap(
            source_patch,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        target = image[ey0:ey1, ex0:ex1]
        weight = (np.exp(-0.90 * radius2) * min(0.82, 0.68 + strength / 500.0)).astype(
            np.float32
        )
        target[:] = _blend(warped, target, weight)


def apply_geometry(
    image: np.ndarray, analysis: FaceAnalysis, state: EditState
) -> np.ndarray:
    """Apply subtle local slimming/chin/eye deformations; preserve the outer canvas.

    Displacements use a bounded padded face ROI, feather to identity outside that area, and
    are capped to protect jaw lines and nearby straight background features. Remapping is
    tiled by rows so a 24–48 MP photo does not require two full-canvas float maps. Inverse
    mapping avoids holes/duplicated pixels common with forward splatting.
    """
    shape_active = any(
        (
            state.face_slim,
            state.jaw_slim,
            state.cheek_slim,
            state.chin_length,
            state.chin_width,
        )
    )
    if not analysis.faces or not (shape_active or state.eye_size):
        return image
    height, width = image.shape[:2]
    output = image.copy()
    for face in analysis.faces:
        rect = _local_rect(face, width, height)
        x0, y0, x1, y1 = rect
        fx, fy, fw, fh = face.pixel_bbox(width, height)
        if fw < 8 or fh < 8 or x1 <= x0 or y1 <= y0:
            continue
        if shape_active:
            # A stable source snapshot lets each remap tile sample untouched neighbor rows.
            source_patch = output[y0:y1, x0:x1].copy()
            rh, rw = source_patch.shape[:2]
            gx = np.arange(x0, x1, dtype=np.float32)[None, :]
            u = (gx - (fx + 0.5 * fw)) / max(1.0, fw)
            abs_u = np.abs(u)
            side = np.sign(u)
            x_region = _smoothstep((abs_u - 0.16) / 0.27) * (
                1.0 - _smoothstep((abs_u - 0.73) / 0.18)
            )
            edge_x = 1.0 - _smoothstep((abs_u - 0.49) / 0.19)
            cheek_value = state.cheek_slim / 100.0
            jaw_value = state.jaw_slim / 100.0
            face_value = state.face_slim / 100.0
            chin_width_value = state.chin_width / 100.0
            chin_amount = (state.chin_length / 50.0) * 0.055
            chin_x_weight = np.clip(1.0 - np.abs(u) / 0.48, 0.0, 1.0)
            anchor = fy + 0.70 * fh
            tile_height = max(24, min(192, 1_000_000 // max(1, rw)))
            destination = output[y0:y1, x0:x1]
            for row0 in range(0, rh, tile_height):
                row1 = min(rh, row0 + tile_height)
                gy = np.arange(y0 + row0, y0 + row1, dtype=np.float32)[:, None]
                v = (gy - fy) / max(1.0, fh)
                cheek_band = np.exp(-0.5 * ((v - 0.59) / 0.22) ** 2)
                jaw_band = np.exp(-0.5 * ((v - 0.79) / 0.22) ** 2)
                chin_band = np.exp(-0.5 * ((v - 0.88) / 0.20) ** 2)
                displacement = fw * (
                    0.030 * face_value * (0.62 * cheek_band + 0.85 * jaw_band)
                    + 0.027 * jaw_value * jaw_band
                    + 0.024 * cheek_value * cheek_band
                    + 0.026 * chin_width_value * chin_band
                )
                displacement = np.minimum(displacement, fw * 0.075)
                source_x = gx + side * displacement * x_region
                chin_y_weight = _smoothstep((v - 0.68) / 0.25)
                source_y = anchor + (gy - anchor) / (
                    1.0 + chin_amount * chin_x_weight * chin_y_weight
                )
                map_x = np.ascontiguousarray(
                    np.broadcast_to(source_x - x0, (row1 - row0, rw)), dtype=np.float32
                )
                map_y = np.ascontiguousarray(
                    np.broadcast_to(source_y - y0, (row1 - row0, rw)), dtype=np.float32
                )
                source_tile = source_patch[row0:row1]
                warped = cv2.remap(
                    source_patch,
                    map_x,
                    map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                edge_y = _smoothstep((v + 0.16) / 0.11) * (
                    1.0 - _smoothstep((v - 1.04) / 0.11)
                )
                feather = (edge_x * edge_y).astype(np.float32)
                destination[row0:row1] = _blend(warped, source_tile, feather)
        if state.eye_size:
            _eye_warp(output, face, rect, state.eye_size)
    return output
