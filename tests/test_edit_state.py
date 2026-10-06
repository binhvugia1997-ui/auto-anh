from glowup_local.models.edit_state import PARAMETER_LIMITS, EditState
from glowup_local.models.presets import available_presets, make_preset, preset_values


def test_preset_loading_and_independent_states():
    assert available_presets() == ("Original", "Natural", "Beauty", "Glow", "Max")
    beauty = make_preset("Beauty")
    assert beauty.preset == "Beauty"
    assert beauty.skin_smooth == preset_values("Beauty")["skin_smooth"]
    beauty.skin_smooth = 0
    assert make_preset("Beauty").skin_smooth != 0
    assert make_preset("Original").master_strength == 0


def test_parameter_limits_and_unknown_preset_fallback():
    state = EditState.from_dict(
        {
            "master_strength": 999,
            "chin_length": -1000,
            "temperature": 900,
            "preset": "not-a-look",
        }
    )
    assert state.master_strength == 100
    assert state.chin_length == -50
    assert state.temperature == 100
    assert state.preset == "Original"
    for name, (low, high) in PARAMETER_LIMITS.items():
        assert low <= getattr(state, name) <= high


def test_state_serialization_roundtrip_and_unknown_keys():
    original = EditState.from_dict(
        {
            "preset": "Glow",
            "master_strength": 73,
            "eye_size": 24,
            "auto_white_balance": True,
            "not_an_edit": "ignored",
        }
    )
    restored = EditState.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    assert "not_an_edit" not in restored.to_dict()
    restored.eye_size = 500
    assert restored.eye_size == 100
