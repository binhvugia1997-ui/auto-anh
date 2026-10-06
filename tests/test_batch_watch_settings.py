import time
from pathlib import Path

import numpy as np
from PIL import Image

from glowup_local.models.edit_state import EditState
from glowup_local.services.batch import BatchResult, process_batch
from glowup_local.services.settings import SettingsStore
from glowup_local.services.watch_folder import (
    DuplicateGuard,
    WatchFolderService,
    file_identity,
)


def _write_photo(path: Path, color=(115, 150, 190)) -> None:
    pixels = np.full((28, 36, 3), color, dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def test_batch_isolates_failures_and_exports_full_resolution(tmp_path: Path):
    broken = tmp_path / "bad.jpg"
    broken.write_bytes(b"not an image")
    valid = tmp_path / "portrait.png"
    _write_photo(valid)
    output = tmp_path / "exports"
    progress = []
    results = process_batch(
        [broken, valid],
        EditState(preset="Original", master_strength=0),
        output,
        extension=".png",
        progress=lambda done, total, item: progress.append((done, total, item)),
    )
    assert len(results) == 2
    assert not results[0].succeeded and results[0].error
    assert results[1].succeeded
    assert results[1].output is not None
    with Image.open(results[1].output) as saved:
        assert saved.size == (36, 28)
    assert len(progress) == 2 and progress[-1][:2] == (2, 2)


def test_watch_duplicate_guard_claims_each_version_once(tmp_path: Path):
    photo = tmp_path / "portrait.jpg"
    identity = file_identity(photo, 100, 12345)
    guard = DuplicateGuard()
    assert guard.claim(identity)
    assert not guard.claim(identity)
    assert not guard.claim(identity)
    assert guard.claim(file_identity(photo, 101, 12346))


def test_watch_folder_waits_for_new_stable_photo_and_stops(tmp_path: Path):
    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    service = WatchFolderService(poll_interval=0.1, stable_checks=2, minimum_age=0)
    results: list[BatchResult] = []
    statuses: list[str] = []
    service.start(
        input_folder,
        output_folder,
        EditState(master_strength=0),
        suffix="beauty",
        extension=".jpg",
        on_status=statuses.append,
        on_item=results.append,
    )
    # A post-start file is a new arrival. The service needs repeated unchanged stats before open.
    _write_photo(input_folder / "portrait.jpg")
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and not results:
        time.sleep(0.05)
    service.stop(wait=True, timeout=5)
    assert len(results) == 1 and results[0].succeeded
    assert results[0].output is not None and results[0].output.exists()
    assert service.processed_count == 1
    assert service.last_status == "Stopped"
    assert "Stopped" in statuses


def test_watch_same_folder_and_windows_unsafe_suffix_do_not_reingest_exports(
    tmp_path: Path,
):
    folder = tmp_path / "camera"
    folder.mkdir()
    service = WatchFolderService(poll_interval=0.1, stable_checks=2, minimum_age=0)
    results: list[BatchResult] = []
    service.start(
        folder,
        folder,
        EditState(master_strength=0),
        suffix="portrait/beauty:*",
        extension=".jpg",
        on_item=results.append,
    )
    _write_photo(folder / "new portrait.jpg")
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and not results:
        time.sleep(0.05)
    time.sleep(0.35)  # give a second scan a chance to notice its own output
    service.stop(wait=True, timeout=5)
    assert len(results) == 1 and results[0].succeeded
    assert len(list(folder.glob("*_portrait_beauty.jpg"))) == 1
    assert service.processed_count == 1


def test_settings_store_roundtrip_and_bounds(tmp_path: Path):
    store = SettingsStore(tmp_path / "config" / "settings.json")
    store.update(
        {
            "last_preset": "Glow",
            "beauty_strength": 240,
            "jpeg_quality": -5,
            "watch_input_folder": "D:/Camera",
        }
    )
    store.save()
    restored = SettingsStore(store.path)
    assert restored.get("last_preset") == "Glow"
    assert restored.get("beauty_strength") == 100
    assert restored.get("jpeg_quality") == 1
    assert restored.get("watch_input_folder") == "D:/Camera"
