"""Race-resistant local watch-folder processing using stable-file polling."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ..image_io import (
    SUPPORTED_INPUT_SUFFIXES,
    load_image,
    output_path_for,
    sanitize_suffix,
    save_image_unique,
)
from ..models.edit_state import EditState
from ..processing.pipeline import render_image
from .batch import BatchResult

logger = logging.getLogger(__name__)
StatusCallback = Callable[[str], None]
ItemCallback = Callable[[BatchResult], None]


def file_identity(
    path: str | os.PathLike[str], size: int, mtime_ns: int
) -> tuple[str, int, int]:
    """Stable duplicate key; normcase provides Windows case-insensitive path semantics."""
    return (os.path.normcase(str(Path(path).resolve())), int(size), int(mtime_ns))


class DuplicateGuard:
    """Remember processed file versions. A changed file is a new version, same version is not."""

    def __init__(self) -> None:
        self._seen: set[tuple[str, int, int]] = set()
        self._lock = threading.Lock()

    def claim(self, identity: tuple[str, int, int]) -> bool:
        with self._lock:
            if identity in self._seen:
                return False
            self._seen.add(identity)
            return True

    def contains(self, identity: tuple[str, int, int]) -> bool:
        with self._lock:
            return identity in self._seen


class WatchFolderService:
    """Poll a local folder, wait for file writes to settle, then process each version once."""

    def __init__(
        self,
        *,
        poll_interval: float = 0.8,
        stable_checks: int = 2,
        minimum_age: float = 1.2,
    ) -> None:
        self.poll_interval = max(0.1, float(poll_interval))
        self.stable_checks = max(2, int(stable_checks))
        self.minimum_age = max(0.0, float(minimum_age))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._guard = DuplicateGuard()
        self._on_status: StatusCallback | None = None
        self._on_item: ItemCallback | None = None
        self._count_lock = threading.Lock()
        self.processed_count = 0
        self.failed_count = 0
        self.last_status = "Stopped"

    @property
    def is_running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop.is_set()
        )

    def start(
        self,
        input_folder: str | os.PathLike[str],
        output_folder: str | os.PathLike[str],
        state: EditState,
        *,
        quality: int = 95,
        suffix: str = "beauty",
        extension: str = ".jpg",
        on_status: StatusCallback | None = None,
        on_item: ItemCallback | None = None,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("The watch folder is already running.")
        source = Path(input_folder).expanduser().resolve()
        destination = Path(output_folder).expanduser().resolve()
        suffix = sanitize_suffix(suffix) or "beauty"
        if not source.is_dir():
            raise NotADirectoryError(f"Input folder does not exist: {source}")
        destination.mkdir(parents=True, exist_ok=True)
        if not os.access(destination, os.W_OK):
            raise PermissionError(f"Output folder is not writable: {destination}")
        self._stop = threading.Event()
        self._on_status, self._on_item = on_status, on_item
        self.processed_count = 0
        self.failed_count = 0
        # Capture a baseline cutoff in the UI thread, then build it lazily in the worker. A
        # camera photo arriving after Watch starts has a newer mtime and is still processed.
        # Back off slightly for coarse FAT/exFAT/network timestamp resolution; a photo created
        # immediately after Start can otherwise appear to predate the session by a few ms.
        session_start_ns = time.time_ns() - 2_000_000_000
        state_snapshot = state.copy()
        self.last_status = f"Watching {source}"
        self._emit_status(self.last_status)
        self._thread = threading.Thread(
            target=self._run,
            args=(
                source,
                destination,
                state_snapshot,
                int(quality),
                suffix,
                extension,
                session_start_ns,
            ),
            name="GlowUpWatchFolder",
            daemon=True,
        )
        self._thread.start()
        logger.info("Watch folder started: input=%s output=%s", source, destination)

    def detach_callbacks(self) -> None:
        """Disconnect UI callbacks before application shutdown if a render is still active."""
        self._on_status = None
        self._on_item = None

    def stop(self, *, wait: bool = False, timeout: float | None = None) -> None:
        self._stop.set()
        self.last_status = "Stopping after current photo…"
        self._emit_status(self.last_status)
        thread = self._thread
        if wait and thread and thread is not threading.current_thread():
            thread.join(timeout)
        if thread is None or not thread.is_alive():
            self.last_status = "Stopped"
            self._emit_status(self.last_status)

    def _emit_status(self, text: str) -> None:
        if self._on_status:
            try:
                self._on_status(text)
            except Exception:
                logger.exception("Watch-folder status callback failed")

    def _emit_item(self, result: BatchResult) -> None:
        if self._on_item:
            try:
                self._on_item(result)
            except Exception:
                logger.exception("Watch-folder item callback failed")

    @staticmethod
    def _is_input_candidate(
        path: Path, source: Path, destination: Path, suffix: str
    ) -> bool:
        if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES or path.name.startswith(
            "."
        ):
            return False
        try:
            resolved = path.resolve()
            # Never re-ingest exports when output is nested below input. Output may equal input;
            # in that case the suffix check below separates generated files from new originals.
            if destination != source and destination in resolved.parents:
                return False
        except OSError:
            return False
        cleaned_suffix = sanitize_suffix(suffix).casefold()
        is_generated = bool(
            cleaned_suffix
            and re.search(
                r"_" + re.escape(cleaned_suffix) + r"(?:_\d+)?$",
                path.stem.casefold(),
            )
        )
        return not is_generated

    def _run(
        self,
        source: Path,
        destination: Path,
        state: EditState,
        quality: int,
        suffix: str,
        extension: str,
        session_start_ns: int,
    ) -> None:
        # Each candidate must have an unchanged size and mtime for multiple polls and be old
        # enough to avoid opening a camera/phone file while its writer is still flushing it.
        stable: dict[str, tuple[int, int, int]] = {}
        try:
            while not self._stop.is_set():
                now = time.time()
                try:
                    paths = list(source.rglob("*"))
                except OSError as exc:
                    logger.warning("Watch-folder scan failed: %s", exc)
                    paths = []
                for path in paths:
                    if self._stop.is_set():
                        break
                    if not path.is_file() or not self._is_input_candidate(
                        path, source, destination, suffix
                    ):
                        continue
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    identity = file_identity(path, stat.st_size, stat.st_mtime_ns)
                    key = os.path.normcase(str(path.resolve()))
                    if stat.st_mtime_ns <= session_start_ns:
                        # Existing files are only claimed, never reprocessed on startup.
                        self._guard.claim(identity)
                        stable.pop(key, None)
                        continue
                    if self._guard.contains(identity):
                        stable.pop(key, None)
                        continue
                    previous = stable.get(key)
                    if (
                        previous
                        and previous[0] == stat.st_size
                        and previous[1] == stat.st_mtime_ns
                    ):
                        count = previous[2] + 1
                    else:
                        count = 1
                    stable[key] = (stat.st_size, stat.st_mtime_ns, count)
                    if (
                        count < self.stable_checks
                        or now - stat.st_mtime < self.minimum_age
                    ):
                        continue
                    # Claim before processing. If decoding fails, do not loop on a corrupt file.
                    if not self._guard.claim(identity):
                        continue
                    try:
                        # Check that the file remained unchanged at the handoff boundary, decode it
                        # once, then validate its signature again before rendering.
                        latest = path.stat()
                        if (
                            latest.st_size != stat.st_size
                            or latest.st_mtime_ns != stat.st_mtime_ns
                        ):
                            continue
                        document = load_image(path)
                        latest = path.stat()
                        if (
                            latest.st_size != stat.st_size
                            or latest.st_mtime_ns != stat.st_mtime_ns
                        ):
                            continue
                        rendered = render_image(document.pixels, state.copy())
                        output_path = output_path_for(
                            path, destination, suffix, extension
                        )
                        output_path = save_image_unique(
                            rendered,
                            output_path,
                            quality=quality,
                            exif=document.exif,
                            icc_profile=document.icc_profile,
                        )
                        result = BatchResult(path, output_path, None)
                    except Exception as exc:
                        logger.exception("Watch-folder processing failed: %s", path)
                        result = BatchResult(path, None, str(exc))
                    with self._count_lock:
                        if result.succeeded:
                            self.processed_count += 1
                        else:
                            self.failed_count += 1
                    self._emit_item(result)
                    logger.info(
                        "Watch-folder item %s: %s",
                        "exported" if result.succeeded else "failed",
                        path,
                    )
                    stable.pop(key, None)
                    if not self._stop.is_set():
                        self.last_status = f"Watching · {self.processed_count} processed · {self.failed_count} failed"
                        self._emit_status(self.last_status)
                self._stop.wait(self.poll_interval)
        finally:
            self.last_status = "Stopped"
            self._emit_status(self.last_status)
            logger.info("Watch folder stopped")
