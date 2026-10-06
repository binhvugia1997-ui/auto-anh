"""Failure-isolated full-resolution batch processing."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from ..image_io import ImageDocument, load_image, output_path_for, save_image_unique
from ..models.edit_state import EditState
from ..processing.pipeline import render_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BatchResult:
    source: Path
    output: Path | None
    error: str | None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.output is not None


ProgressCallback = Callable[[int, int, BatchResult], None]


def process_batch(
    files: Iterable[str | Path],
    state: EditState,
    output_folder: str | Path,
    *,
    quality: int = 95,
    suffix: str = "beauty",
    extension: str = ".jpg",
    progress: ProgressCallback | None = None,
) -> list[BatchResult]:
    """Process each source independently; one bad photo never aborts the batch."""
    sources = [Path(path) for path in files]
    results: list[BatchResult] = []
    total = len(sources)
    for index, source in enumerate(sources, start=1):
        try:
            document: ImageDocument = load_image(source)
            rendered = render_image(document.pixels, state.copy())
            destination = output_path_for(source, output_folder, suffix, extension)
            destination = save_image_unique(
                rendered,
                destination,
                quality=quality,
                exif=document.exif,
                icc_profile=document.icc_profile,
            )
            result = BatchResult(source, destination, None)
            logger.info("Batch export complete: %s -> %s", source, destination)
        except (
            Exception
        ) as exc:  # Failure is scoped to this input, including permissions/decoding.
            logger.exception("Batch item failed: %s", source)
            result = BatchResult(source, None, str(exc))
        results.append(result)
        if progress:
            progress(index, total, result)
    return results
