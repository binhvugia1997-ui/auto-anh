"""Safe image loading, preview sizing, and non-overwriting export helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_INPUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_OUTPUT_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImageDocument:
    """One immutable-in-practice source image and the metadata needed for export."""

    path: Path
    pixels: np.ndarray  # RGB or RGBA, uint8, EXIF-oriented
    exif: bytes | None = None
    icc_profile: bytes | None = None
    source_format: str | None = None

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


def is_supported_image(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_INPUT_SUFFIXES


def load_image(path: str | os.PathLike[str]) -> ImageDocument:
    """Decode an image with EXIF orientation applied; never return a mutable PIL view."""
    file_path = Path(path).expanduser()
    with Image.open(file_path) as source:
        source_format = source.format
        icc_profile = source.info.get("icc_profile")
        exif_data = source.getexif()
        oriented = ImageOps.exif_transpose(source)
        # Preserve transparency for PNG/WebP; CMYK, palette and grayscale are normalized.
        has_alpha = oriented.mode in {"RGBA", "LA"} or (
            oriented.mode == "P" and "transparency" in oriented.info
        )
        converted = oriented.convert("RGBA" if has_alpha else "RGB")
        pixels = np.asarray(converted, dtype=np.uint8).copy()
        if exif_data:
            # Pixels are now physically oriented, so exported EXIF must not rotate them again.
            exif_data[274] = 1
            exif_bytes = exif_data.tobytes()
        else:
            exif_bytes = None
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("This image could not be converted to RGB or RGBA.")
    return ImageDocument(file_path, pixels, exif_bytes, icc_profile, source_format)


def load_thumbnail(
    path: str | os.PathLike[str], max_dimension: int = 100
) -> np.ndarray:
    """Decode a small oriented thumbnail without retaining a full-resolution NumPy copy."""
    if max_dimension < 1:
        raise ValueError("max_dimension must be positive")
    with Image.open(Path(path).expanduser()) as source:
        oriented = ImageOps.exif_transpose(source)
        has_alpha = oriented.mode in {"RGBA", "LA"} or (
            oriented.mode == "P" and "transparency" in oriented.info
        )
        thumbnail = oriented.convert("RGBA" if has_alpha else "RGB")
        thumbnail.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        return np.asarray(thumbnail, dtype=np.uint8).copy()


def make_preview(image: np.ndarray, max_dimension: int = 1600) -> np.ndarray:
    """Downscale for interaction while preserving aspect ratio and channel count."""
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("Cannot preview an empty or malformed image.")
    if max_dimension < 1:
        raise ValueError("max_dimension must be positive")
    height, width = image.shape[:2]
    longest = max(width, height)
    if longest <= max_dimension:
        return np.ascontiguousarray(image.copy())
    scale = max_dimension / float(longest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    pil = Image.fromarray(image)
    preview = pil.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(preview, dtype=np.uint8).copy()


def sanitize_suffix(suffix: str) -> str:
    """Normalize a user-provided filename suffix for Windows-safe output names."""
    invalid_filename_chars = set('<>:"/|?*') | {chr(92)}
    cleaned = "".join(
        "_" if char in invalid_filename_chars or ord(char) < 32 else char
        for char in str(suffix).strip()
    )
    return cleaned.strip("._ ")[:80].rstrip(". ")


def output_path_for(
    source_path: str | os.PathLike[str],
    output_folder: str | os.PathLike[str],
    suffix: str = "beauty",
    extension: str | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Create a safe destination name, adding _2, _3, ... for collisions."""
    source = Path(source_path)
    out_dir = Path(output_folder).expanduser()
    ext = (extension or source.suffix or ".jpg").lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported export format: {ext}")
    clean_suffix = sanitize_suffix(suffix)
    stem = source.stem if not clean_suffix else f"{source.stem}_{clean_suffix}"
    if stem.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        stem = "_" + stem
    candidate = out_dir / f"{stem}{ext}"
    if overwrite or not candidate.exists():
        return candidate
    number = 2
    while True:
        candidate = out_dir / f"{stem}_{number}{ext}"
        if not candidate.exists():
            return candidate
        number += 1


def _to_pil(image: np.ndarray, file_format: str) -> Image.Image:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Export expects a uint8 RGB or RGBA image.")
    pil = Image.fromarray(np.ascontiguousarray(image))
    if file_format == "JPEG":
        if pil.mode == "RGBA":
            background = Image.new("RGB", pil.size, (255, 255, 255))
            background.paste(pil, mask=pil.getchannel("A"))
            pil = background
        elif pil.mode != "RGB":
            pil = pil.convert("RGB")
    return pil


def save_image_unique(
    image: np.ndarray,
    desired_path: str | os.PathLike[str],
    *,
    quality: int = 95,
    exif: bytes | None = None,
    icc_profile: bytes | None = None,
) -> Path:
    """Atomically claim a destination and retry with a numbered suffix on a race."""
    desired = Path(desired_path).expanduser()
    for number in range(1, 10000):
        candidate = (
            desired
            if number == 1
            else desired.with_name(f"{desired.stem}_{number}{desired.suffix}")
        )
        try:
            return save_image(
                image, candidate, quality=quality, exif=exif, icc_profile=icc_profile
            )
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not find a free export name for {desired.name}")


def save_image(
    image: np.ndarray,
    path: str | os.PathLike[str],
    *,
    quality: int = 95,
    exif: bytes | None = None,
    icc_profile: bytes | None = None,
    overwrite: bool = False,
) -> Path:
    """Save an image without silently replacing an existing file.

    When overwrite is false, an exclusive create prevents a race between naming and saving.
    The source photo is never touched by this function unless a caller explicitly opts in.
    """
    destination = Path(path).expanduser()
    file_format = SUPPORTED_OUTPUT_FORMATS.get(destination.suffix.lower())
    if file_format is None:
        raise ValueError("Choose a JPG or PNG export filename.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    quality = min(100, max(1, int(quality)))
    pil = _to_pil(image, file_format)
    save_options: dict[str, object] = {}
    if file_format == "JPEG":
        save_options["quality"] = quality
        save_options["optimize"] = True
        save_options["subsampling"] = 0
    else:
        save_options["compress_level"] = 6
    if exif:
        save_options["exif"] = exif
    if icc_profile:
        save_options["icc_profile"] = icc_profile

    if overwrite:
        pil.save(destination, format=file_format, **save_options)
        return destination

    descriptor: int | None = None
    try:
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        with os.fdopen(descriptor, "wb") as output:  # type: ignore[arg-type]
            descriptor = None
            pil.save(output, format=file_format, **save_options)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError:
        raise
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except OSError as cleanup_error:
            logger.debug(
                "Could not remove incomplete export %s: %s", destination, cleanup_error
            )
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return destination
