# GlowUp Local

GlowUp Local is a Windows 10/11 desktop photo editor for private, local portrait retouching. It imports photos, detects faces, renders an interactive preview, and exports a new full-resolution file. Originals are never edited in place.

## Requirements

- Windows 10/11 x64
- Python 3.11 or newer
- Internet access only to install Python dependencies (not needed for editing after setup)
- About 1 GB free RAM for ordinary phone photos; large 24–48 MP images benefit from more memory

The runtime uses PySide6, NumPy, OpenCV and Pillow. There are no paid services, API keys, photo uploads, telemetry, or runtime model downloads. OpenCV's bundled Haar cascades work offline. For a denser optional local face mesh, see **Optional MediaPipe landmarks** below.

## Install and run (Windows)

1. Install 64-bit Python 3.11+ and enable the Python launcher (`py`).
2. From this project folder, run `setup.bat` once.
3. Run `run.bat`.

Equivalent commands:

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m glowup_local
```

For tests, install `requirements-dev.txt` and run `python -m pytest -q` (or `test.bat`). On Linux, PySide6 also needs the platform's Qt/OpenGL/XKB system libraries and a display; on headless machines use `QT_QPA_PLATFORM=offscreen` if those libraries are installed.

## Editing workflow

1. **Open Photos** or drop JPG, JPEG, PNG, WEBP, BMP or TIFF files onto the preview. The queue keeps paths and small thumbnails; only the selected full-resolution source is held in memory.
2. GlowUp detects faces locally and immediately renders the selected look at preview resolution. Choose Original, Natural, Beauty, Glow or Max. **AUTO BEAUTY** applies the balanced Beauty look in one click.
3. Adjust the grouped sliders on the right. Interactive rendering is debounced and performed in a worker thread on a preview (longest edge 1,500 px). Zoom with the wheel, pan by dragging while zoomed, hold **Before**, or use Split.
4. Export Photo renders from the full-resolution source into the configured output folder. Export settings include JPG/PNG, JPEG quality, folder and filename suffix. Existing destinations receive `_2`, `_3`, and so on; source files are not overwritten.

Presets are defined centrally in `glowup_local/models/presets.py`. Beauty Strength blends the rendered look against the exact original. Undo/redo stores edit-state snapshots rather than full-resolution bitmaps. Slider updates are grouped into one history action per drag.

### Batch processing

Import several files, choose the look/strength and export options, then click **Process All**. The batch uses a snapshot of the current photo's state (or the last selected preset if no photo is selected), renders each file at its source resolution, reports progress, and isolates decode/write failures per photo.

### Watch Folder

Click **Watch Folder**, choose an input and output directory, and enable watching. Existing files are treated as a baseline. New files must have an unchanged size/mtime across multiple polls and be old enough to be considered finished before processing starts. Each file version is attempted once per watcher session; failed and completed items are logged. Stop Watching stops new work and lets the current photo finish. If output is nested under input, generated output files are excluded from rescanning.

## Features

- Dark, resizable PySide6 UI with photo queue, preview, zoom/pan, before/after and split comparison
- Local face/eye detection, soft face/skin/eye/lip/teeth masks, face/jaw/cheek/chin deformation, skin retouching, eye/lip/teeth controls, color/light and detail controls
- Natural, Beauty, Glow and Max looks, adjustable overall strength, undo/redo
- Background preview, export and batch jobs; full-resolution export and EXIF orientation correction
- JPG/PNG output, collision-safe filenames, useful EXIF/ICC retention where supported
- Optional polling Watch Folder, local JSON settings and rotating logs
- Unicode paths/names (including Vietnamese filenames); corrupt files and no-face images fail gracefully

## Architecture

```text
glowup_local/
  main.py                 QApplication entry point
  image_io.py             EXIF-aware loading, preview scaling and safe export
  models/                  bounded EditState and centralized presets
  face/detection.py        optional MediaPipe + offline OpenCV face detection
  processing/              local geometry, masks and RGB processing pipeline
  services/                batch, watch-folder, settings and rotating logs
  ui/                      Qt main window, canvas, dialogs and QRunnable workers
tests/                     offline unit/integration tests
run.bat, setup.bat, test.bat
```

### Optional MediaPipe landmarks

The core app has no mandatory model download. Its offline Haar fallback estimates the face contour and feature centers. To enable MediaPipe Face Mesh, install the optional vision dependency:

```bat
.venv\Scripts\python.exe -m pip install -r requirements-vision.txt
```

MediaPipe's installed wheel includes its local Face Mesh model assets; GlowUp does not fetch a model at startup. When the optional runtime is importable, the app uses its dense landmarks for face, eye and lip masks; otherwise it falls back to OpenCV Haar cascades. MediaPipe is optional because its dependency set is substantially larger than the core editor.

## Keyboard shortcuts

- `Ctrl+O` — open photos
- `Ctrl+S` — export selected photo
- `Ctrl+Z` — undo
- `Ctrl+Shift+Z` — redo
- Hold `Space` — show the original while comparing (when focus is not in a text/slider control)
- Mouse wheel — zoom; right-click the preview or click **Fit** to reset zoom

## Settings, logs and privacy

Settings and rotating diagnostics are stored in the user's application-data directory (`%APPDATA%\GlowUpLocal` on Windows; `~/.config/glowup-local` on Linux). Logs are under its `logs` subdirectory. Logs contain paths and processing/error status, never image pixels. No photo bytes are sent to a network service. The only outbound network activity required is package installation; the optional MediaPipe model assets are included in its installed package.

## Tests

```bat
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

Tests cover preset/state bounds and serialization, Unicode image load/save, EXIF orientation, preview sizing, output collisions, no-face processing, alpha preservation, local geometry/masks, batch failure isolation, watch duplicate prevention and settings persistence. Tests are fully offline.

## Known limitations

- Haar cascades are not a dense landmark model. Without the optional MediaPipe runtime, facial feature locations are estimated from the detected rectangle; rotated/occluded/profile faces and small/distant faces may be missed. Hair, eyelid and tooth masks remain soft geometric/color estimates rather than semantic segmentation. Inspect the preview before exporting.
- Face slimming and chin/eye warps are deliberately capped and locally feathered; some background pixels immediately beside a face can still move slightly. Max is not a substitute for a professional liquify/mesh editor.
- Skin tone, lips and teeth controls use conservative color/mask heuristics. They are not intended to replace manual retouching on every lighting/skin-tone combination.
- Interactive previews are downscaled; the final render uses the full-resolution source and re-runs local face detection. Tiny preview/full-resolution differences may occur.
- Very large images and bilateral filtering can use substantial memory and take longer than a preview render. Batch and Watch Folder are sequential by design to bound memory; a long-running photo is allowed to finish when Watch Folder is stopped.
- ICC profiles and useful EXIF are retained when Pillow supports them; some format-specific metadata, maker notes and unsupported chunks may not survive conversion. PNG transparency is retained; JPEG transparency is composited onto white.

## Troubleshooting

- **`run.bat` says the environment is missing:** run `setup.bat` from the project folder.
- **Python launcher missing:** install Python 3.11+ x64 and enable `py` on PATH, or create `.venv` with your Python executable and install `requirements.txt`.
- **A photo cannot be read:** check that it is a supported image and not corrupt or still being written. The technical reason is in the local log.
- **No face found:** color/detail adjustments still work. Try a larger, frontal, well-lit image; optionally install the local MediaPipe runtime.
- **Export permission/disk error:** choose a writable output folder with enough free space. The source is left untouched.
- **Missing Qt platform/OpenGL libraries on Linux:** install your distribution's PySide6 runtime dependencies and use a graphical session (or an offscreen Qt platform for headless tests).
