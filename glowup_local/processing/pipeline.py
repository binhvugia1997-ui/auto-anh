"""Local, non-destructive portrait and color pipeline (RGB in, RGB(A) out)."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from ..face.detection import FaceAnalysis, FaceRegion, detect_faces
from ..models.edit_state import EditState
from .geometry import apply_geometry
from .masks import FaceMasks, masks_for_face


def _blend_region(
    target: np.ndarray, edited: np.ndarray, mask: np.ndarray, opacity: float = 1.0
) -> np.ndarray:
    alpha = np.ascontiguousarray(
        mask.astype(np.float32) * (max(0.0, min(1.0, opacity)) / 255.0)
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    return cv2.blendLinear(edited, target, alpha, 1.0 - alpha)


def _lut_rgb(
    image: np.ndarray, red: np.ndarray, green: np.ndarray, blue: np.ndarray
) -> np.ndarray:
    table = np.stack((red, green, blue), axis=1).reshape(256, 1, 3).astype(np.uint8)
    return cv2.LUT(image, table)


def _simple_lut(function) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    return np.clip(np.rint(function(x)), 0, 255).astype(np.uint8)


def _apply_rgb_bias(image: np.ndarray, bias: tuple[float, float, float]) -> np.ndarray:
    tables = [_simple_lut(lambda x, amount=delta: x + amount) for delta in bias]
    return _lut_rgb(image, *tables)


def _crop(working: np.ndarray, masks: FaceMasks) -> tuple[slice, slice, np.ndarray]:
    x0, y0, x1, y1 = masks.rect
    return slice(y0, y1), slice(x0, x1), working[y0:y1, x0:x1]


def _protect_skin(
    adjusted: np.ndarray,
    original: np.ndarray,
    faces: Iterable[FaceRegion],
    amount: float,
) -> None:
    """Dampen global local-contrast/sharpening inside soft skin masks, not eye detail."""
    if amount <= 0:
        return
    for face in faces:
        masks = masks_for_face(adjusted.shape, face)
        ys, xs, _ = _crop(adjusted, masks)
        target = adjusted[ys, xs]
        source = original[ys, xs]
        target[:] = _blend_region(target, source, masks.skin, amount)


def _apply_facial_edits(
    working: np.ndarray, faces: Iterable[FaceRegion], state: EditState
) -> np.ndarray:
    facial_active = any(
        (
            state.skin_smooth,
            state.skin_brightness,
            state.blemish_reduction,
            state.under_eye_reduction,
            state.skin_tone,
            state.eye_brightness,
            state.dark_circle_reduction,
            state.eye_sharpness,
            state.lip_color,
            state.lip_saturation,
            state.teeth_whitening,
        )
    )
    if not facial_active:
        return working
    for face in faces:
        masks = masks_for_face(working.shape, face)
        ys, xs, patch_view = _crop(working, masks)
        patch = patch_view.copy()
        skin_mask = masks.skin

        # Bilateral smoothing is edge-aware, and only the soft skin mask is composited back.
        if state.skin_smooth:
            smooth = cv2.bilateralFilter(
                patch,
                d=7,
                sigmaColor=22 + state.skin_smooth * 0.36,
                sigmaSpace=8 + state.skin_smooth * 0.16,
            )
            patch = _blend_region(
                patch, smooth, skin_mask, 0.68 * state.skin_smooth / 100.0
            )
        if state.blemish_reduction:
            # A tiny median neighborhood suppresses isolated blemish noise without flattening
            # the full face; the lower blend amount retains pores and fine facial texture.
            median = cv2.medianBlur(patch, 5)
            patch = _blend_region(
                patch, median, skin_mask, 0.42 * state.blemish_reduction / 100.0
            )
        if state.skin_brightness:
            gamma = 1.0 - 0.17 * state.skin_brightness / 100.0
            lift = cv2.LUT(
                patch,
                _simple_lut(lambda x, value=gamma: 255.0 * np.power(x / 255.0, value)),
            )
            patch = _blend_region(patch, lift, skin_mask, 0.55)
        if state.skin_tone:
            amount = state.skin_tone / 100.0
            tint = (
                (18 * amount, 2 * amount, -18 * amount)
                if amount > 0
                else (15 * amount, -1 * amount, -15 * amount)
            )
            toned = _apply_rgb_bias(patch, tint)
            patch = _blend_region(patch, toned, skin_mask, 0.18)

        if state.under_eye_reduction or state.dark_circle_reduction:
            level = max(state.under_eye_reduction, state.dark_circle_reduction)
            gamma = 1.0 - 0.22 * level / 100.0
            lifted = cv2.LUT(
                patch,
                _simple_lut(lambda x, value=gamma: 255.0 * np.power(x / 255.0, value)),
            )
            amount = (
                0.45 * state.under_eye_reduction / 100.0
                + 0.42 * state.dark_circle_reduction / 100.0
            )
            patch = _blend_region(patch, lifted, masks.under_eyes, min(0.75, amount))
        if state.eye_brightness:
            gamma = 1.0 - 0.27 * state.eye_brightness / 100.0
            bright = cv2.LUT(
                patch,
                _simple_lut(lambda x, value=gamma: 255.0 * np.power(x / 255.0, value)),
            )
            patch = _blend_region(patch, bright, masks.eyes, 0.54)
        if state.eye_sharpness:
            blurred = cv2.GaussianBlur(patch, (0, 0), sigmaX=0.8)
            sharp = cv2.addWeighted(
                patch,
                1.0 + 0.85 * state.eye_sharpness / 100.0,
                blurred,
                -0.85 * state.eye_sharpness / 100.0,
                0,
            )
            patch = _blend_region(patch, sharp, masks.eyes, 0.75)

        if state.lip_color:
            # A muted rose overlay is blended through a soft lip ellipse, never painted solid.
            target = np.empty_like(patch)
            target[:] = (178, 79, 104)
            patch = _blend_region(
                patch, target, masks.lips, 0.28 * state.lip_color / 100.0
            )
        if state.lip_saturation:
            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            hsv = hsv.copy()
            sat = hsv[..., 1].astype(np.float32)
            hsv[..., 1] = np.clip(
                sat * (1.0 + 0.55 * state.lip_saturation / 100.0), 0, 255
            ).astype(np.uint8)
            more_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            patch = _blend_region(patch, more_color, masks.lips, 0.72)
        if state.teeth_whitening:
            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            hsv_work = hsv.copy()
            value = hsv_work[..., 2].astype(np.float32)
            saturation = hsv_work[..., 1].astype(np.float32)
            # Whitening is luminance-limited: darker, saturated lip pixels receive less effect.
            selectivity = np.clip((value - 55.0) / 125.0, 0, 1) * np.clip(
                (175.0 - saturation) / 125.0, 0, 1
            )
            hsv_work[..., 1] = np.clip(
                saturation * (1 - 0.48 * state.teeth_whitening / 100.0), 0, 255
            ).astype(np.uint8)
            hsv_work[..., 2] = np.clip(
                value + 20 * state.teeth_whitening / 100.0, 0, 255
            ).astype(np.uint8)
            whitened = cv2.cvtColor(hsv_work, cv2.COLOR_HSV2RGB)
            teeth_mask = (masks.teeth.astype(np.float32) * selectivity).astype(np.uint8)
            patch = _blend_region(patch, whitened, teeth_mask, 0.78)

        working[ys, xs] = patch
    return working


def _sample_rgb(image: np.ndarray, max_dimension: int = 512) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / float(max(height, width)))
    if scale == 1:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _apply_color(image: np.ndarray, state: EditState) -> np.ndarray:
    working = image
    if state.auto_white_balance:
        sample = _sample_rgb(working)
        means = np.mean(sample.reshape(-1, 3), axis=0).astype(np.float64)
        gray = float(np.mean(means))
        gains = np.clip(gray / np.maximum(means, 1.0), 0.78, 1.28)
        luts = [_simple_lut(lambda x, gain=g: x * gain) for g in gains]
        working = _lut_rgb(working, *luts)
    if state.auto_exposure:
        sample = _sample_rgb(working)
        luma = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
        median = max(1.0, float(np.median(luma)))
        ev = float(np.clip(np.log2(118.0 / median), -0.8, 0.8))
        working = cv2.LUT(working, _simple_lut(lambda x: x * (2.0**ev)))

    exposure_ev = 3.0 * state.exposure / 100.0
    brightness = 62.0 * state.brightness / 100.0
    contrast = 1.0 + state.contrast / 100.0
    highlights = 34.0 * state.highlights / 100.0
    shadows = 34.0 * state.shadows / 100.0
    if any(
        (
            state.exposure,
            state.brightness,
            state.contrast,
            state.highlights,
            state.shadows,
        )
    ):

        def tone_curve(values: np.ndarray) -> np.ndarray:
            unit = values / 255.0
            curved = values * (2.0**exposure_ev)
            curved += brightness
            curved = (curved - 127.5) * contrast + 127.5
            curved += shadows * np.power(1.0 - unit, 1.7)
            curved += highlights * np.power(unit, 1.7)
            return curved

        tone_lut = _simple_lut(tone_curve)
        working = cv2.LUT(working, tone_lut)

    if state.temperature or state.tint:
        temperature = state.temperature / 100.0
        tint = state.tint / 100.0
        luts = (
            _simple_lut(lambda x: x + 34 * temperature + 7 * tint),
            _simple_lut(lambda x: x - 12 * tint),
            _simple_lut(lambda x: x - 34 * temperature + 7 * tint),
        )
        working = _lut_rgb(working, *luts)
    if state.saturation or state.vibrance:
        hsv = cv2.cvtColor(working, cv2.COLOR_RGB2HSV)
        hsv = hsv.copy()
        sat = hsv[..., 1].astype(np.float32)
        value = sat * (1.0 + state.saturation / 100.0)
        value += (255.0 - sat) * (0.55 * state.vibrance / 100.0)
        hsv[..., 1] = np.clip(value, 0, 255).astype(np.uint8)
        working = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return working


def _detail(
    image: np.ndarray, state: EditState, faces: Iterable[FaceRegion]
) -> np.ndarray:
    working = image
    if state.noise_reduction:
        amount = state.noise_reduction / 100.0
        denoised = cv2.bilateralFilter(
            working, d=5, sigmaColor=7 + 17 * amount, sigmaSpace=4
        )
        working = cv2.addWeighted(
            working, 1.0 - 0.50 * amount, denoised, 0.50 * amount, 0
        )
    if state.clarity:
        amount = state.clarity / 100.0
        source = working
        low = cv2.GaussianBlur(source, (0, 0), sigmaX=3.0)
        if amount > 0:
            adjusted = cv2.addWeighted(
                source, 1.0 + 0.32 * amount, low, -0.32 * amount, 0
            )
            _protect_skin(adjusted, source, faces, 0.38 * amount)
            working = adjusted
        else:
            working = cv2.addWeighted(
                source, 1.0 - 0.22 * abs(amount), low, 0.22 * abs(amount), 0
            )
    if state.sharpness:
        amount = state.sharpness / 100.0
        source = working
        low = cv2.GaussianBlur(source, (0, 0), sigmaX=0.9)
        adjusted = cv2.addWeighted(source, 1.0 + 0.72 * amount, low, -0.72 * amount, 0)
        _protect_skin(adjusted, source, faces, 0.55 * amount)
        working = adjusted
    return working


def _blend_master_strength(
    processed: np.ndarray, original: np.ndarray, strength: int
) -> np.ndarray:
    if strength >= 100:
        return processed
    if strength <= 0:
        return original.copy()
    # Row chunks cap temporary allocation on 24/48 MP images while preserving exact blending.
    factor = strength / 100.0
    height = processed.shape[0]
    for y0 in range(0, height, 256):
        y1 = min(height, y0 + 256)
        current = cv2.addWeighted(
            processed[y0:y1], factor, original[y0:y1], 1.0 - factor, 0
        )
        processed[y0:y1] = current
    return processed


def render_image(
    image: np.ndarray, state: EditState, faces: FaceAnalysis | None = None
) -> np.ndarray:
    """Render edits from the source pixels; caller-owned input arrays are never modified.

    Face geometry and masks operate on face-local patches. Global operations use OpenCV LUTs
    and native filters. The master strength is a final, chunked blend against the exact source.
    """
    if (
        image is None
        or image.size == 0
        or image.ndim != 3
        or image.shape[2] not in (3, 4)
    ):
        raise ValueError("Expected a non-empty uint8 RGB or RGBA image.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    state.normalize()
    source_rgb = np.ascontiguousarray(image[..., :3])
    if state.master_strength <= 0:
        return image.copy()
    needs_faces = any(
        (
            state.face_slim,
            state.jaw_slim,
            state.cheek_slim,
            state.chin_length,
            state.chin_width,
            state.eye_size,
            state.skin_smooth,
            state.skin_brightness,
            state.blemish_reduction,
            state.under_eye_reduction,
            state.skin_tone,
            state.eye_brightness,
            state.dark_circle_reduction,
            state.eye_sharpness,
            state.lip_color,
            state.lip_saturation,
            state.teeth_whitening,
        )
    )
    analysis = (
        faces
        if faces is not None
        else (detect_faces(source_rgb) if needs_faces else FaceAnalysis())
    )
    working = source_rgb.copy()
    working = apply_geometry(working, analysis, state)
    working = _apply_facial_edits(working, analysis.faces, state)
    working = _apply_color(working, state)
    working = _detail(working, state, analysis.faces)
    working = _blend_master_strength(working, source_rgb, state.master_strength)
    if image.shape[2] == 4:
        result = image.copy()
        result[..., :3] = working
        return result
    return working
