"""Offscreen GUI smoke test; Linux environments without Qt's native runtime skip cleanly."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication

    from glowup_local.ui.main_window import MainWindow
except (
    ImportError
) as exc:  # Minimal CI containers may lack system libGL/XKB/DBus packages.
    pytest.skip(
        f"Qt native runtime unavailable in this environment: {exc}",
        allow_module_level=True,
    )

from glowup_local.services.settings import SettingsStore


def _wait(app: QApplication, predicate, timeout: float = 12.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


def _make_photo(path: Path, offset: int = 0) -> None:
    y, x = np.mgrid[:180, :240]
    pixels = np.stack(
        (
            (x + offset) % 255,
            (y * 2 + offset) % 255,
            np.full_like(x, 125 + offset % 50),
        ),
        axis=2,
    ).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def test_import_presets_slider_history_compare_export_and_batch(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    first = tmp_path / "ảnh một.png"
    second = tmp_path / "portrait two.jpg"
    _make_photo(first)
    _make_photo(second, 28)
    output = tmp_path / "exports"
    store = SettingsStore(tmp_path / "settings.json")
    store.update(
        {
            "last_preset": "Natural",
            "last_output_directory": str(output),
            "filename_suffix": "beauty",
            "output_format": ".jpg",
        }
    )
    window = MainWindow(store)
    window.show()
    window.add_files([str(first)])
    assert _wait(
        app, lambda: window.document is not None and window.last_preview is not None
    )
    assert window.photo_list.count() == 1

    for preset in ("Natural", "Beauty", "Glow", "Max"):
        window.apply_preset(preset)
        assert window.current_entry.state.preset == preset
    face_slider = window.controls["face_slim"]
    face_slider.setValue(24)
    eye_slider = window.controls["eye_size"]
    eye_slider.setValue(15)
    exposure = window.controls["exposure"]
    exposure.setValue(18)
    assert _wait(app, lambda: not window._history_timer.isActive())
    assert window.current_entry.state.face_slim == 24
    assert window.current_entry.state.eye_size == 15
    assert window.current_entry.state.exposure == 18
    window.undo()
    assert window.current_entry.state.exposure != 18
    window.redo()
    assert window.current_entry.state.exposure == 18

    window.canvas.set_mode("before")
    assert window.canvas.mode == "before"
    window.canvas.set_mode("split")
    assert window.canvas.mode == "split"
    window.canvas.set_mode("after")
    assert window.canvas.mode == "after"

    window.export_current()
    assert _wait(app, lambda: not window._export_running)
    assert list(output.glob("*_beauty.jpg"))

    window.add_files([str(second)])
    assert _wait(
        app,
        lambda: (
            window.document is not None
            and window.current_entry is not None
            and window.current_entry.path == second.resolve()
        ),
    )
    assert _wait(app, lambda: window.last_preview is not None)
    window.process_all()
    assert _wait(app, lambda: not window._batch_running, timeout=20)
    assert len(list(output.glob("*.jpg"))) >= 3
    window.close()
    app.processEvents()
