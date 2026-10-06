from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from glowup_local.image_io import (
    load_image,
    load_thumbnail,
    make_preview,
    output_path_for,
    save_image,
    save_image_unique,
)


def test_preview_scaling_preserves_ratio_and_channels():
    image = np.zeros((3000, 2000, 4), dtype=np.uint8)
    preview = make_preview(image, max_dimension=750)
    assert preview.shape == (750, 500, 4)
    assert make_preview(image, 5000).shape == image.shape


def test_unicode_load_save_jpg_png_and_collision(tmp_path: Path):
    source = tmp_path / "ảnh chân dung.jpg"
    source_pixels = np.zeros((37, 61, 3), dtype=np.uint8)
    source_pixels[:] = (82, 143, 208)
    Image.fromarray(source_pixels).save(source, quality=100)
    loaded = load_image(source)
    thumb = load_thumbnail(source, 20)
    assert max(thumb.shape[:2]) == 20
    assert loaded.width == 61 and loaded.height == 37
    assert loaded.pixels.shape == (37, 61, 3)

    output_dir = tmp_path / "kết quả"
    output_dir.mkdir()
    first = output_path_for(source, output_dir, "beauty", ".jpg")
    assert first.name == "ảnh chân dung_beauty.jpg"
    safe_suffix = output_path_for(
        source, output_dir, "folder" + chr(92) + "bad:?*", ".png"
    )
    assert "/" not in safe_suffix.name and chr(92) not in safe_suffix.name
    assert ":" not in safe_suffix.name and "?" not in safe_suffix.name
    save_image(loaded.pixels, first, quality=100, exif=loaded.exif)
    assert load_image(first).pixels.shape == loaded.pixels.shape
    second = output_path_for(source, output_dir, "beauty", ".jpg")
    assert second.name == "ảnh chân dung_beauty_2.jpg"
    with pytest.raises(FileExistsError):
        save_image(loaded.pixels, first, quality=95)
    raced = save_image_unique(loaded.pixels, first, quality=95)
    assert raced == second and raced.exists()

    alpha = np.zeros((13, 19, 4), dtype=np.uint8)
    alpha[..., :3] = (200, 110, 45)
    alpha[..., 3] = 128
    png = output_path_for(source, output_dir, "alpha", ".png")
    save_image(alpha, png)
    assert load_image(png).pixels.shape == alpha.shape
    assert np.array_equal(load_image(png).pixels, alpha)


def test_exif_orientation_is_applied_and_export_orientation_reset(tmp_path: Path):
    source = tmp_path / "oriented.jpg"
    pixels = np.zeros((20, 40, 3), dtype=np.uint8)
    pixels[:, :20] = (255, 0, 0)
    pixels[:, 20:] = (0, 0, 255)
    exif = Image.Exif()
    exif[274] = 6  # rotate 90° clockwise on display
    Image.fromarray(pixels).save(source, exif=exif)
    document = load_image(source)
    assert (document.width, document.height) == (20, 40)
    output = tmp_path / "oriented_beauty.jpg"
    save_image(document.pixels, output, exif=document.exif)
    with Image.open(output) as saved:
        assert saved.getexif().get(274) == 1
        assert saved.size == (20, 40)
