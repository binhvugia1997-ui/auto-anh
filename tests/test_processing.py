import numpy as np

from glowup_local.face.detection import FaceAnalysis, FaceRegion, detect_faces
from glowup_local.models.edit_state import EditState
from glowup_local.processing.geometry import apply_geometry
from glowup_local.processing.masks import masks_for_face
from glowup_local.processing.pipeline import render_image


def test_no_face_image_processes_without_crashing_and_preserves_source():
    yy, xx = np.mgrid[:96, :144]
    image = np.stack(
        (xx * 255 // 143, yy * 255 // 95, np.full_like(xx, 90)), axis=2
    ).astype(np.uint8)
    source = image.copy()
    assert len(detect_faces(image).faces) == 0
    state = EditState(
        preset="Natural",
        master_strength=100,
        exposure=8,
        contrast=7,
        saturation=9,
        skin_smooth=70,
    )
    result = render_image(image, state, FaceAnalysis())
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, image)
    assert np.array_equal(image, source)


def test_master_strength_is_a_final_blend():
    image = np.full((24, 30, 3), (60, 120, 180), dtype=np.uint8)
    full = render_image(
        image, EditState(master_strength=100, exposure=16), FaceAnalysis()
    )
    half = render_image(
        image, EditState(master_strength=50, exposure=16), FaceAnalysis()
    )
    expected = np.rint(
        full.astype(np.float32) * 0.5 + image.astype(np.float32) * 0.5
    ).astype(np.uint8)
    assert np.max(np.abs(half.astype(int) - expected.astype(int))) <= 1
    unchanged = render_image(
        image, EditState(master_strength=0, exposure=80), FaceAnalysis()
    )
    assert np.array_equal(unchanged, image)


def test_local_geometry_is_feathered_and_never_changes_outer_canvas():
    image = np.zeros((220, 320, 3), dtype=np.uint8)
    image[:, :160] = 80
    image[:, 160:] = 190
    face = FaceRegion(
        bbox=(0.32, 0.14, 0.36, 0.64),
        left_eye=(0.43, 0.36),
        right_eye=(0.57, 0.36),
        nose=(0.50, 0.48),
        mouth=(0.50, 0.65),
    )
    state = EditState(master_strength=100, face_slim=70, jaw_slim=40, eye_size=30)
    result = apply_geometry(image, FaceAnalysis((face,)), state)
    assert result.shape == image.shape
    # The affected face ROI is local; distant walls/background remain bit-identical.
    assert np.array_equal(result[:, :45], image[:, :45])
    assert np.array_equal(result[:, 275:], image[:, 275:])
    assert np.count_nonzero(result != image) > 0
    assert np.array_equal(image[:, :45], np.zeros_like(image[:, :45]) + 80)


def test_masks_are_soft_and_region_sized():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    face = FaceRegion(
        (0.25, 0.12, 0.5, 0.72), (0.40, 0.37), (0.60, 0.37), (0.50, 0.52), (0.50, 0.68)
    )
    masks = masks_for_face(image.shape, face)
    x0, y0, x1, y1 = masks.rect
    assert masks.skin.shape == (y1 - y0, x1 - x0)
    assert masks.skin.max() > 0
    assert masks.eyes.max() > 0 and masks.lips.max() > 0 and masks.teeth.max() > 0
    assert np.any((masks.skin > 0) & (masks.skin < 255))


def test_global_sharpening_is_damped_inside_detected_skin():
    rng = np.random.default_rng(7)
    noise = rng.integers(-14, 15, size=(180, 180, 3), dtype=np.int16)
    image = np.clip(120 + noise, 0, 255).astype(np.uint8)
    face = FaceRegion(
        (0.22, 0.08, 0.56, 0.84),
        (0.39, 0.34),
        (0.61, 0.34),
        (0.50, 0.49),
        (0.50, 0.68),
    )
    state = EditState(master_strength=100, sharpness=100)
    unrestricted = render_image(image, state, FaceAnalysis())
    protected = render_image(image, state, FaceAnalysis((face,)))
    masks = masks_for_face(image.shape, face)
    x0, y0, x1, y1 = masks.rect
    skin = masks.skin > 180
    original_skin = image[y0:y1, x0:x1][skin].astype(np.int16)
    unrestricted_skin = unrestricted[y0:y1, x0:x1][skin].astype(np.int16)
    protected_skin = protected[y0:y1, x0:x1][skin].astype(np.int16)
    assert (
        np.abs(protected_skin - original_skin).mean()
        < np.abs(unrestricted_skin - original_skin).mean()
    )
    # Skin protection is face-local; distant background remains as sharpened as before.
    assert np.array_equal(protected[:, :20], unrestricted[:, :20])


def test_alpha_channel_is_preserved():
    image = np.zeros((30, 40, 4), dtype=np.uint8)
    image[..., :3] = (90, 110, 130)
    image[..., 3] = np.arange(40, dtype=np.uint8)[None, :] * 6
    result = render_image(
        image, EditState(master_strength=70, exposure=15), FaceAnalysis()
    )
    assert result.shape == image.shape
    assert np.array_equal(result[..., 3], image[..., 3])
    assert not np.array_equal(result[..., :3], image[..., :3])
