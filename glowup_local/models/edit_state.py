"""The single source of truth for non-destructive edits."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar

PARAMETER_LIMITS: dict[str, tuple[int, int]] = {
    "master_strength": (0, 100),
    "face_slim": (0, 100),
    "jaw_slim": (0, 100),
    "cheek_slim": (0, 100),
    "chin_length": (-50, 50),
    "chin_width": (0, 100),
    "skin_smooth": (0, 100),
    "skin_brightness": (0, 100),
    "blemish_reduction": (0, 100),
    "under_eye_reduction": (0, 100),
    "skin_tone": (-100, 100),
    "eye_size": (0, 100),
    "eye_brightness": (0, 100),
    "dark_circle_reduction": (0, 100),
    "eye_sharpness": (0, 100),
    "lip_color": (0, 100),
    "lip_saturation": (0, 100),
    "teeth_whitening": (0, 100),
    "exposure": (-100, 100),
    "brightness": (-100, 100),
    "contrast": (-100, 100),
    "highlights": (-100, 100),
    "shadows": (-100, 100),
    "temperature": (-100, 100),
    "tint": (-100, 100),
    "saturation": (-100, 100),
    "vibrance": (-100, 100),
    "sharpness": (0, 100),
    "clarity": (-100, 100),
    "noise_reduction": (0, 100),
}

BOOLEAN_FIELDS = ("auto_white_balance", "auto_exposure")
STATE_FIELDS = tuple(PARAMETER_LIMITS) + ("preset",) + BOOLEAN_FIELDS
PRESET_NAMES = ("Original", "Natural", "Beauty", "Glow", "Max")


@dataclass(slots=True)
class EditState:
    """Serializable, bounded edit parameters. Values are deliberately unitless UI values."""

    preset: str = "Original"
    master_strength: int = 0
    face_slim: int = 0
    jaw_slim: int = 0
    cheek_slim: int = 0
    chin_length: int = 0
    chin_width: int = 0
    skin_smooth: int = 0
    skin_brightness: int = 0
    blemish_reduction: int = 0
    under_eye_reduction: int = 0
    skin_tone: int = 0
    eye_size: int = 0
    eye_brightness: int = 0
    dark_circle_reduction: int = 0
    eye_sharpness: int = 0
    lip_color: int = 0
    lip_saturation: int = 0
    teeth_whitening: int = 0
    exposure: int = 0
    brightness: int = 0
    contrast: int = 0
    highlights: int = 0
    shadows: int = 0
    temperature: int = 0
    tint: int = 0
    saturation: int = 0
    vibrance: int = 0
    sharpness: int = 0
    clarity: int = 0
    noise_reduction: int = 0
    auto_white_balance: bool = False
    auto_exposure: bool = False

    LIMITS: ClassVar[dict[str, tuple[int, int]]] = PARAMETER_LIMITS

    def __setattr__(self, name: str, value: Any) -> None:
        # Keep bounds true even for callers that edit a state directly rather than through UI.
        if name in PARAMETER_LIMITS:
            low, high = PARAMETER_LIMITS[name]
            try:
                value = min(high, max(low, round(float(value))))
            except (TypeError, ValueError, OverflowError):
                value = 0
        elif name == "preset" and value not in PRESET_NAMES:
            value = "Original"
        elif name in BOOLEAN_FIELDS:
            value = bool(value)
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self.normalize()

    def normalize(self) -> EditState:
        if self.preset not in PRESET_NAMES:
            self.preset = "Original"
        for name, (low, high) in PARAMETER_LIMITS.items():
            raw = getattr(self, name)
            try:
                value = round(float(raw))
            except (TypeError, ValueError, OverflowError):
                value = 0
            setattr(self, name, min(high, max(low, value)))
        for name in BOOLEAN_FIELDS:
            setattr(self, name, bool(getattr(self, name)))
        return self

    def copy(self, **changes: Any) -> EditState:
        values = self.to_dict()
        values.update(changes)
        return EditState.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EditState:
        data = data or {}
        valid = {field.name for field in fields(cls)}
        values = {key: value for key, value in data.items() if key in valid}
        return cls(**values)
