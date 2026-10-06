"""Small, local JSON settings store with atomic writes and conservative defaults."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


def user_data_directory() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return root / "GlowUpLocal"
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "glowup-local"


class SettingsStore:
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "last_open_directory": str(Path.home()),
        "last_output_directory": str(Path.home() / "Pictures" / "GlowUp"),
        "last_preset": "Natural",
        "beauty_strength": 100,
        "jpeg_quality": 95,
        "output_format": ".jpg",
        "filename_suffix": "beauty",
        "watch_input_folder": "",
        "watch_output_folder": "",
    }

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else user_data_directory() / "settings.json"
        self.values = dict(self.DEFAULTS)
        self.load()

    def load(self) -> dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as source:
                stored = json.load(source)
            if isinstance(stored, dict):
                self.values.update(stored)
        except FileNotFoundError:
            logger.debug("No existing settings file at %s", self.path)
        except (OSError, json.JSONDecodeError) as exc:
            # Settings corruption should not prevent an application launch.
            logger.warning("Could not read settings %s: %s", self.path, exc)
        try:
            self.values["beauty_strength"] = min(
                100, max(0, int(self.values.get("beauty_strength", 100)))
            )
        except (TypeError, ValueError, OverflowError):
            self.values["beauty_strength"] = int(self.DEFAULTS["beauty_strength"])
        try:
            self.values["jpeg_quality"] = min(
                100, max(1, int(self.values.get("jpeg_quality", 95)))
            )
        except (TypeError, ValueError, OverflowError):
            self.values["jpeg_quality"] = int(self.DEFAULTS["jpeg_quality"])
        if self.values.get("output_format") not in {".jpg", ".png"}:
            self.values["output_format"] = ".jpg"
        return dict(self.values)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, self.DEFAULTS.get(key, default))

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def update(self, values: dict[str, Any]) -> None:
        self.values.update(values)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            prefix="settings-", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as destination:
                json.dump(self.values, destination, indent=2, ensure_ascii=False)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(name, self.path)
        except Exception:
            try:
                os.unlink(name)
            except OSError as cleanup_error:
                logger.debug(
                    "Could not remove temporary settings file %s: %s",
                    name,
                    cleanup_error,
                )
            raise
