"""QThreadPool tasks; workers only return values and never touch widgets."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal

from ..face.detection import detect_faces
from ..image_io import (
    ImageDocument,
    load_image,
    load_thumbnail,
    make_preview,
    save_image_unique,
)
from ..models.edit_state import EditState
from ..processing.pipeline import render_image
from ..services.batch import process_batch

logger = logging.getLogger(__name__)


class LoadSignals(QObject):
    loaded = Signal(
        str, object, object, object, object
    )  # path, doc, preview, faces, error


class LoadWorker(QRunnable):
    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = str(path)
        self.signals = LoadSignals()

    def run(self) -> None:
        try:
            document = load_image(self.path)
            preview = make_preview(document.pixels, 1500)
            faces = detect_faces(preview)
            self.signals.loaded.emit(self.path, document, preview, faces, None)
        except Exception as exc:
            logger.exception("Image import failed: %s", self.path)
            self.signals.loaded.emit(self.path, None, None, None, str(exc))


class ThumbnailSignals(QObject):
    ready = Signal(str, object, object)


class ThumbnailWorker(QRunnable):
    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = str(path)
        self.signals = ThumbnailSignals()

    def run(self) -> None:
        try:
            thumbnail = load_thumbnail(self.path, 100)
            self.signals.ready.emit(self.path, thumbnail, None)
        except Exception as exc:  # noqa: BLE001 - corrupt/unsupported photos are per-item errors
            logger.warning("Thumbnail decode failed for %s: %s", self.path, exc)
            self.signals.ready.emit(self.path, None, str(exc))


class PreviewSignals(QObject):
    rendered = Signal(int, object, object)  # token, pixels, error


class PreviewWorker(QRunnable):
    def __init__(
        self, token: int, source: object, state: EditState, faces: object
    ) -> None:
        super().__init__()
        self.token = token
        self.source = source
        self.state = state.copy()
        self.faces = faces
        self.signals = PreviewSignals()

    def run(self) -> None:
        try:
            output = render_image(self.source, self.state, self.faces)
            self.signals.rendered.emit(self.token, output, None)
        except Exception as exc:
            logger.exception("Preview render failed")
            self.signals.rendered.emit(self.token, None, str(exc))


class ExportSignals(QObject):
    finished = Signal(object, object)  # path, error


class ExportWorker(QRunnable):
    def __init__(
        self, document: ImageDocument, state: EditState, output: Path, quality: int
    ) -> None:
        super().__init__()
        self.document = document
        self.state = state.copy()
        self.output = output
        self.quality = quality
        self.signals = ExportSignals()

    def run(self) -> None:
        try:
            rendered = render_image(self.document.pixels, self.state)
            destination = save_image_unique(
                rendered,
                self.output,
                quality=self.quality,
                exif=self.document.exif,
                icc_profile=self.document.icc_profile,
            )
            logger.info("Export complete: %s", destination)
            self.signals.finished.emit(destination, None)
        except Exception as exc:
            logger.exception("Export failed: %s", self.output)
            self.signals.finished.emit(None, str(exc))


class BatchSignals(QObject):
    progress = Signal(int, int, object)
    finished = Signal(object)


class BatchWorker(QRunnable):
    def __init__(
        self,
        files: list[str],
        state: EditState,
        output_folder: Path,
        quality: int,
        suffix: str,
        extension: str,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.state = state.copy()
        self.output_folder = output_folder
        self.quality = quality
        self.suffix = suffix
        self.extension = extension
        self.signals = BatchSignals()

    def run(self) -> None:
        try:
            results = process_batch(
                self.files,
                self.state,
                self.output_folder,
                quality=self.quality,
                suffix=self.suffix,
                extension=self.extension,
                progress=lambda done, total, item: self.signals.progress.emit(
                    done, total, item
                ),
            )
        except Exception as exc:
            logger.exception("Batch failed before item processing")
            results = [exc]
        self.signals.finished.emit(results)
