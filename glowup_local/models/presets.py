"""Central preset definitions; all preset controls map to EditState fields."""

from __future__ import annotations

from .edit_state import PRESET_NAMES, EditState

# Values are intentionally conservative. The sliders, not hidden code, define each look.
_PRESET_VALUES: dict[str, dict[str, int | bool]] = {
    "Original": {
        "master_strength": 0,
    },
    "Natural": {
        "master_strength": 100,
        "skin_smooth": 14,
        "skin_brightness": 9,
        "blemish_reduction": 12,
        "under_eye_reduction": 12,
        "eye_brightness": 6,
        "dark_circle_reduction": 10,
        "teeth_whitening": 4,
        "exposure": 3,
        "shadows": 5,
        "vibrance": 4,
        "sharpness": 7,
        "noise_reduction": 4,
    },
    "Beauty": {
        "master_strength": 100,
        "face_slim": 8,
        "jaw_slim": 5,
        "cheek_slim": 5,
        "skin_smooth": 32,
        "skin_brightness": 17,
        "blemish_reduction": 28,
        "under_eye_reduction": 28,
        "eye_brightness": 14,
        "eye_sharpness": 9,
        "dark_circle_reduction": 25,
        "lip_saturation": 10,
        "teeth_whitening": 10,
        "exposure": 5,
        "shadows": 8,
        "vibrance": 9,
        "sharpness": 12,
        "noise_reduction": 9,
    },
    "Glow": {
        "master_strength": 100,
        "face_slim": 8,
        "jaw_slim": 5,
        "cheek_slim": 4,
        "skin_smooth": 37,
        "skin_brightness": 27,
        "blemish_reduction": 32,
        "under_eye_reduction": 32,
        "eye_brightness": 20,
        "eye_sharpness": 12,
        "dark_circle_reduction": 30,
        "lip_saturation": 14,
        "teeth_whitening": 13,
        "exposure": 10,
        "highlights": -4,
        "shadows": 12,
        "temperature": 3,
        "vibrance": 15,
        "sharpness": 13,
        "noise_reduction": 10,
    },
    "Max": {
        "master_strength": 100,
        "face_slim": 23,
        "jaw_slim": 17,
        "cheek_slim": 17,
        "skin_smooth": 62,
        "skin_brightness": 36,
        "blemish_reduction": 55,
        "under_eye_reduction": 55,
        "eye_size": 14,
        "eye_brightness": 27,
        "eye_sharpness": 20,
        "dark_circle_reduction": 48,
        "lip_saturation": 20,
        "teeth_whitening": 23,
        "exposure": 11,
        "shadows": 16,
        "vibrance": 21,
        "sharpness": 18,
        "noise_reduction": 17,
    },
}


def available_presets() -> tuple[str, ...]:
    return PRESET_NAMES


def preset_values(name: str) -> dict[str, int | bool]:
    if name not in _PRESET_VALUES:
        raise ValueError(f"Unknown preset: {name}")
    return dict(_PRESET_VALUES[name])


def make_preset(name: str) -> EditState:
    """Return a fresh state with preset controls reset to the centrally defined values."""
    return EditState(preset=name, **preset_values(name))
