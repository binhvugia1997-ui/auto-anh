"""Main GlowUp Local photo editor window."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import (
    QIcon,
    QKeySequence,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..face.detection import FaceAnalysis
from ..image_io import (
    ImageDocument,
    is_supported_image,
    output_path_for,
)
from ..models.edit_state import EditState
from ..models.presets import available_presets, make_preset
from ..services.batch import BatchResult
from ..services.settings import SettingsStore
from ..services.watch_folder import WatchFolderService
from .image_canvas import ImageCanvas
from .watch_dialog import WatchFolderDialog
from .widgets import SectionHeader, SliderControl
from .workers import (
    BatchWorker,
    ExportWorker,
    LoadWorker,
    PreviewWorker,
    ThumbnailWorker,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PhotoEntry:
    path: Path
    state: EditState
    faces: FaceAnalysis | None = None
    undo_stack: list[dict] = field(default_factory=list)
    redo_stack: list[dict] = field(default_factory=list)

    @property
    def key(self) -> str:
        return os.path.normcase(str(self.path.resolve()))


class UiBridge(QObject):
    watch_status = Signal(str)
    watch_item = Signal(object)


class MainWindow(QMainWindow):
    """Photo editor; full-resolution sources remain immutable until export."""

    _SLIDERS = (
        (
            "master_strength",
            "Beauty Strength",
            0,
            100,
            "Blend the selected look with the untouched original.",
        ),
        (
            "skin_smooth",
            "Skin Smooth",
            0,
            100,
            "Edge-aware smoothing limited to the estimated skin region.",
        ),
        (
            "skin_brightness",
            "Skin Brightness",
            0,
            100,
            "Gently lift skin luminance without whitening the whole photo.",
        ),
        (
            "blemish_reduction",
            "Blemish Reduction",
            0,
            100,
            "Reduce isolated marks while retaining natural skin texture.",
        ),
        (
            "under_eye_reduction",
            "Under-eye Reduction",
            0,
            100,
            "Soften under-eye shadows inside a feathered eye mask.",
        ),
        (
            "skin_tone",
            "Skin Tone",
            -100,
            100,
            "Negative values cool the skin; positive values warm it slightly.",
        ),
        (
            "face_slim",
            "Slim Face",
            0,
            100,
            "Local cheek/jaw inward warp with a soft neighborhood falloff.",
        ),
        (
            "jaw_slim",
            "Jaw Slim",
            0,
            100,
            "Subtle inward movement focused on the lower jaw.",
        ),
        (
            "cheek_slim",
            "Cheek Slim",
            0,
            100,
            "Subtle inward movement around the cheek contour.",
        ),
        (
            "chin_length",
            "Chin Length",
            -50,
            50,
            "Negative shortens and positive lengthens the lower chin region.",
        ),
        (
            "chin_width",
            "Chin Width",
            0,
            100,
            "Narrow the lower chin locally; zero leaves the source unchanged.",
        ),
        (
            "eye_size",
            "Eye Size",
            0,
            100,
            "Subtle radial eye enlargement, feathered around each detected eye.",
        ),
        (
            "eye_brightness",
            "Eye Brightness",
            0,
            100,
            "Lift the eye region without painting over lashes.",
        ),
        (
            "dark_circle_reduction",
            "Dark Circle Reduction",
            0,
            100,
            "Reduce under-eye darkness with soft local brightening.",
        ),
        (
            "eye_sharpness",
            "Eye Sharpness",
            0,
            100,
            "Add restrained edge detail within the eye mask.",
        ),
        (
            "lip_color",
            "Lip Color",
            0,
            100,
            "Blend a muted rose tint into the estimated lip region.",
        ),
        (
            "lip_saturation",
            "Lip Saturation",
            0,
            100,
            "Adjust color intensity within the lip mask.",
        ),
        (
            "teeth_whitening",
            "Teeth Whitening",
            0,
            100,
            "Reduce yellow/saturated tones and lift likely tooth pixels gently.",
        ),
        (
            "exposure",
            "Exposure",
            -100,
            100,
            "Exposure compensation, up to approximately ±3 EV.",
        ),
        (
            "brightness",
            "Brightness",
            -100,
            100,
            "Offset image brightness without changing source pixels.",
        ),
        ("contrast", "Contrast", -100, 100, "Adjust contrast around middle gray."),
        (
            "highlights",
            "Highlights",
            -100,
            100,
            "Recover or lift bright tonal regions.",
        ),
        ("shadows", "Shadows", -100, 100, "Lift or deepen dark tonal regions."),
        (
            "temperature",
            "Temperature",
            -100,
            100,
            "Shift the white balance cooler or warmer.",
        ),
        ("tint", "Tint", -100, 100, "Shift the white balance toward green or magenta."),
        (
            "saturation",
            "Saturation",
            -100,
            100,
            "Adjust color intensity across the image.",
        ),
        (
            "vibrance",
            "Vibrance",
            -100,
            100,
            "Preferentially adjust less-saturated colors.",
        ),
        ("sharpness", "Sharpness", 0, 100, "Edge-aware unsharp-mask detail."),
        (
            "clarity",
            "Clarity",
            -100,
            100,
            "Adjust local contrast; negative values soften it.",
        ),
        (
            "noise_reduction",
            "Noise Reduction",
            0,
            100,
            "Edge-preserving bilateral noise reduction.",
        ),
    )

    def __init__(self, settings: SettingsStore | None = None, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings or SettingsStore()
        preset = self.settings.get("last_preset", "Natural")
        if preset not in available_presets():
            preset = "Natural"
        self.default_state = make_preset(preset)
        # The strength preference follows the selected preset across new imports.
        if preset != "Original":
            self.default_state.master_strength = int(
                self.settings.get("beauty_strength", 100)
            )
        self.default_state.normalize()
        self.entries: dict[str, PhotoEntry] = {}
        self.document: ImageDocument | None = None
        self.current_entry: PhotoEntry | None = None
        self.current_key: str | None = None
        self.preview_source: np.ndarray | None = None
        self.preview_faces: FaceAnalysis | None = None
        self.last_preview: np.ndarray | None = None
        self._updating_controls = False
        self._history_before: dict | None = None
        self._preview_revision = 0
        self._preview_inflight = False
        self._preview_pending = False
        self._batch_running = False
        self._export_running = False
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(
            max(2, min(4, QThreadPool.globalInstance().maxThreadCount()))
        )
        self._history_timer = QTimer(self)
        self._history_timer.setSingleShot(True)
        self._history_timer.setInterval(420)
        self._history_timer.timeout.connect(self._commit_history)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(130)
        self._preview_timer.timeout.connect(self._dispatch_preview)
        self._watcher = WatchFolderService()
        self._bridge = UiBridge(self)
        self._bridge.watch_status.connect(self._on_watch_status)
        self._bridge.watch_item.connect(self._on_watch_item)
        self._build_ui()
        self._set_shortcuts()
        self._sync_controls()
        self._set_document_controls(False)
        self.setWindowTitle("GlowUp Local")
        self.resize(1400, 900)
        self.setMinimumSize(960, 680)
        self.statusBar().showMessage("Ready · photos are processed on this device")
        logger.info("GlowUp Local started")

    def _build_ui(self) -> None:
        self.setObjectName("mainWindow")
        self.setStyleSheet(self._stylesheet())
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(12, 10, 12, 8)
        central_layout.setSpacing(9)
        central_layout.addWidget(self._build_header())
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self._build_queue_panel())
        self.splitter.addWidget(self._build_preview_panel())
        self.splitter.addWidget(self._build_controls_panel())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([205, 780, 310])
        central_layout.addWidget(self.splitter, 1)
        central_layout.addWidget(self._build_bottom_panel())
        self.setCentralWidget(central)
        status = QStatusBar(self)
        status.setSizeGripEnabled(False)
        self.setStatusBar(status)

    @staticmethod
    def _stylesheet() -> str:
        return """
        QWidget { background:#171c23; color:#e6eaf0; font-family:'Segoe UI',Arial,sans-serif; font-size:9pt; }
        QMainWindow#mainWindow { background:#12161c; }
        QFrame#panel { background:#1a2028; border:1px solid #29313c; border-radius:11px; }
        QLabel#brand { color:#f6f7f9; font-size:16pt; font-weight:700; letter-spacing:1px; }
        QLabel#subtleText { color:#8995a4; font-size:8pt; }
        QLabel#errorText { color:#ff9d92; }
        QLabel#fileTitle { color:#f1f4f8; font-weight:600; font-size:10pt; }
        QLabel#sectionHeader { color:#8491a2; font-size:8pt; font-weight:700; letter-spacing:1.2px; padding:9px 0 3px; border-bottom:1px solid #2b333e; }
        QLabel#controlLabel { color:#d7dde5; font-size:8.5pt; }
        QLabel#controlValue { color:#aeb9c7; font-size:8.5pt; min-width:34px; }
        QPushButton { background:#252d37; color:#e8edf4; border:1px solid #343e4b; border-radius:7px; padding:7px 11px; }
        QPushButton:hover { background:#303a47; border-color:#4a596c; }
        QPushButton:pressed { background:#202731; }
        QPushButton:disabled { background:#20252c; color:#687382; border-color:#2b3139; }
        QPushButton#primaryButton { background:#7c5cff; border-color:#8f75ff; color:white; font-weight:700; padding:10px 18px; }
        QPushButton#primaryButton:hover { background:#8d72ff; }
        QPushButton#exportButton { background:#2b9b76; border-color:#32aa82; color:white; font-weight:700; padding:10px 18px; }
        QPushButton#exportButton:hover { background:#35ad85; }
        QPushButton#quietButton { background:transparent; border-color:#343d49; }
        QPushButton#presetButton { background:#1e252e; padding:7px 15px; }
        QPushButton#presetButton:checked { background:#312b4d; border-color:#8a72ff; color:#e5deff; }
        QPushButton#toggleButton:checked { background:#29473e; border-color:#3b9773; }
        QListWidget { background:#141920; border:1px solid #29313c; border-radius:8px; outline:0; padding:4px; }
        QListWidget::item { border-radius:6px; padding:4px; color:#d6dce4; }
        QListWidget::item:selected { background:#302b48; color:white; }
        QListWidget::item:hover:!selected { background:#222a34; }
        QSlider::groove:horizontal { height:4px; background:#353e49; border-radius:2px; }
        QSlider::sub-page:horizontal { background:#8b73ff; border-radius:2px; }
        QSlider::handle:horizontal { background:#f0edff; border:2px solid #8970ff; width:12px; height:12px; margin:-5px 0; border-radius:7px; }
        QLineEdit, QComboBox, QSpinBox { background:#12171d; border:1px solid #343e49; border-radius:6px; padding:6px 8px; selection-background-color:#7056e7; }
        QComboBox::drop-down { border:0; width:20px; }
        QProgressBar { background:#222a33; color:#dce2e9; border:1px solid #313a45; border-radius:5px; text-align:center; height:12px; }
        QProgressBar::chunk { background:#8b73ff; border-radius:4px; }
        QScrollArea { border:0; background:transparent; }
        QScrollBar:vertical { background:#171c23; width:9px; margin:2px; }
        QScrollBar::handle:vertical { background:#3c4653; min-height:24px; border-radius:4px; }
        QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
        QSplitter::handle { background:#12161c; width:7px; }
        QStatusBar { background:#12161c; color:#9ba6b4; border-top:1px solid #252c36; }
        QToolTip { color:#f0f2f5; background:#252d38; border:1px solid #475363; padding:5px; }
        """

    def _build_header(self) -> QWidget:
        header = QWidget()
        row = QHBoxLayout(header)
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(10)
        brand_icon = QLabel("✦")
        brand_icon.setStyleSheet("color:#a48fff;font-size:22pt;font-weight:700;")
        brand_icon.setFixedWidth(28)
        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        brand = QLabel("GlowUp Local")
        brand.setObjectName("brand")
        subtitle = QLabel("PRIVATE PORTRAIT EDITOR")
        subtitle.setObjectName("subtleText")
        title_col.addWidget(brand)
        title_col.addWidget(subtitle)
        row.addWidget(brand_icon)
        row.addLayout(title_col)
        row.addStretch(1)
        self.header_file_label = QLabel("No photo selected")
        self.header_file_label.setObjectName("subtleText")
        row.addWidget(self.header_file_label)
        self.open_button = QPushButton("＋  Open Photos")
        self.open_button.setObjectName("quietButton")
        self.open_button.clicked.connect(self.open_photos)
        row.addWidget(self.open_button)
        return header

    def _build_queue_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setMinimumWidth(165)
        panel.setMaximumWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)
        top = QHBoxLayout()
        heading = QLabel("PHOTO QUEUE")
        heading.setStyleSheet(
            "color:#aeb8c5;font-weight:700;letter-spacing:1px;font-size:8pt;"
        )
        top.addWidget(heading, 1)
        self.photo_count_label = QLabel("0")
        self.photo_count_label.setObjectName("subtleText")
        top.addWidget(self.photo_count_label)
        layout.addLayout(top)
        self.photo_list = QListWidget()
        self.photo_list.setIconSize(QSize(52, 52))
        self.photo_list.setSpacing(3)
        self.photo_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.photo_list.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.photo_list, 1)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("＋ Add")
        self.add_button.clicked.connect(self.open_photos)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setObjectName("quietButton")
        self.remove_button.clicked.connect(self.remove_selected)
        buttons.addWidget(self.add_button, 1)
        buttons.addWidget(self.remove_button)
        layout.addLayout(buttons)
        note = QLabel("Drop photos onto the preview to add them.")
        note.setObjectName("subtleText")
        note.setWordWrap(True)
        layout.addWidget(note)
        return panel

    def _build_preview_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 9)
        layout.setSpacing(7)
        title_row = QHBoxLayout()
        self.preview_file_label = QLabel("Preview")
        self.preview_file_label.setObjectName("fileTitle")
        title_row.addWidget(self.preview_file_label, 1)
        self.face_status_label = QLabel("Add a photo to begin")
        self.face_status_label.setObjectName("subtleText")
        title_row.addWidget(self.face_status_label)
        layout.addLayout(title_row)
        self.canvas = ImageCanvas()
        self.canvas.files_dropped.connect(self.add_files)
        layout.addWidget(self.canvas, 1)
        footer = QHBoxLayout()
        self.preview_status_label = QLabel(
            "Wheel to zoom · drag to pan · hold Before to compare"
        )
        self.preview_status_label.setObjectName("subtleText")
        footer.addWidget(self.preview_status_label, 1)
        self.zoom_reset_button = QPushButton("Fit")
        self.zoom_reset_button.setObjectName("quietButton")
        self.zoom_reset_button.setFixedWidth(54)
        self.zoom_reset_button.clicked.connect(self.canvas.reset_view)
        footer.addWidget(self.zoom_reset_button)
        layout.addLayout(footer)
        return panel

    def _build_controls_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setMinimumWidth(250)
        panel.setMaximumWidth(390)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        header = QLabel("  FINE-TUNE")
        header.setStyleSheet(
            "color:#bec8d4;font-weight:700;letter-spacing:1px;padding:13px 10px 9px;"
        )
        outer.addWidget(header)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(13, 0, 13, 14)
        content_layout.setSpacing(4)
        self.controls: dict[str, SliderControl] = {}
        sections = {
            "BEAUTY": [
                "master_strength",
                "skin_smooth",
                "skin_brightness",
                "blemish_reduction",
                "under_eye_reduction",
                "skin_tone",
            ],
            "FACE SHAPE": [
                "face_slim",
                "jaw_slim",
                "cheek_slim",
                "chin_length",
                "chin_width",
            ],
            "EYES": [
                "eye_size",
                "eye_brightness",
                "dark_circle_reduction",
                "eye_sharpness",
            ],
            "LIPS & TEETH": ["lip_color", "lip_saturation", "teeth_whitening"],
            "LIGHT & COLOR": [
                "exposure",
                "brightness",
                "contrast",
                "highlights",
                "shadows",
                "temperature",
                "tint",
                "saturation",
                "vibrance",
            ],
            "DETAIL": ["sharpness", "clarity", "noise_reduction"],
        }
        labels = {item[0]: item[1:] for item in self._SLIDERS}
        for section, names in sections.items():
            content_layout.addWidget(SectionHeader(section))
            for name in names:
                label, low, high, tooltip = labels[name]
                control = SliderControl(label, low, high, 0, tooltip)
                control.valueChanged.connect(
                    lambda value, key=name: self._control_changed(key, value)
                )
                control.sliderPressed.connect(self._begin_history)
                control.sliderReleased.connect(self._commit_history)
                self.controls[name] = control
                content_layout.addWidget(control)
            if section == "LIGHT & COLOR":
                auto_row = QHBoxLayout()
                auto_row.setContentsMargins(0, 2, 0, 6)
                self.auto_wb_button = QPushButton("Auto White Balance")
                self.auto_wb_button.setObjectName("toggleButton")
                self.auto_wb_button.setCheckable(True)
                self.auto_wb_button.setToolTip(
                    "Estimate neutral channel gains locally from this photo."
                )
                self.auto_wb_button.clicked.connect(self._toggle_auto_wb)
                self.auto_exposure_button = QPushButton("Auto Exposure")
                self.auto_exposure_button.setObjectName("toggleButton")
                self.auto_exposure_button.setCheckable(True)
                self.auto_exposure_button.setToolTip(
                    "Set exposure from a downscaled local luminance sample."
                )
                self.auto_exposure_button.clicked.connect(self._toggle_auto_exposure)
                auto_row.addWidget(self.auto_wb_button, 1)
                auto_row.addWidget(self.auto_exposure_button, 1)
                content_layout.addLayout(auto_row)
        content_layout.addWidget(SectionHeader("EXPORT"))
        folder_row = QHBoxLayout()
        self.output_folder_edit = QLineEdit(
            str(self.settings.get("last_output_directory", ""))
        )
        self.output_folder_edit.setPlaceholderText("Output folder")
        self.output_folder_edit.setToolTip(
            "Exports go here. Originals are never replaced by default."
        )
        self.output_folder_edit.editingFinished.connect(self._output_settings_changed)
        self.output_browse_button = QPushButton("Browse…")
        self.output_browse_button.clicked.connect(self._browse_output_folder)
        folder_row.addWidget(self.output_folder_edit, 1)
        folder_row.addWidget(self.output_browse_button)
        content_layout.addLayout(folder_row)
        export_options = QHBoxLayout()
        self.suffix_edit = QLineEdit(
            str(self.settings.get("filename_suffix", "beauty"))
        )
        self.suffix_edit.setPlaceholderText("Filename suffix")
        self.suffix_edit.setToolTip("Output example: portrait_beauty.jpg")
        self.suffix_edit.editingFinished.connect(self._output_settings_changed)
        self.format_combo = QComboBox()
        self.format_combo.addItem("JPG", ".jpg")
        self.format_combo.addItem("PNG", ".png")
        saved_format = self.settings.get("output_format", ".jpg")
        saved_index = self.format_combo.findData(saved_format)
        if saved_index >= 0:
            self.format_combo.setCurrentIndex(saved_index)
        self.format_combo.currentIndexChanged.connect(self._output_settings_changed)
        export_options.addWidget(self.suffix_edit, 1)
        export_options.addWidget(self.format_combo)
        content_layout.addLayout(export_options)
        quality_row = QHBoxLayout()
        quality_label = QLabel("JPEG quality")
        quality_label.setObjectName("controlLabel")
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(int(self.settings.get("jpeg_quality", 95)))
        self.quality_spin.setSuffix(" / 100")
        self.quality_spin.valueChanged.connect(self._output_settings_changed)
        quality_row.addWidget(quality_label, 1)
        quality_row.addWidget(self.quality_spin)
        content_layout.addLayout(quality_row)
        self.export_hint = QLabel(
            "PNG preserves transparency · JPG uses the quality setting"
        )
        self.export_hint.setObjectName("subtleText")
        self.export_hint.setWordWrap(True)
        content_layout.addWidget(self.export_hint)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        return panel

    def _build_bottom_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        preset_row = QHBoxLayout()
        preset_row.setSpacing(7)
        look_label = QLabel("LOOKS")
        look_label.setStyleSheet(
            "font-weight:700;color:#8f9baa;letter-spacing:1px;font-size:8pt;"
        )
        preset_row.addWidget(look_label)
        self.preset_group = QButtonGroup(self)
        self.preset_group.setExclusive(True)
        self.preset_buttons: dict[str, QPushButton] = {}
        for name in available_presets():
            button = QPushButton(name)
            button.setObjectName("presetButton")
            button.setCheckable(True)
            button.setToolTip(
                f"Apply the {name} preset and update the editing controls."
            )
            button.clicked.connect(
                lambda checked=False, preset_name=name: self.apply_preset(preset_name)
            )
            self.preset_group.addButton(button)
            self.preset_buttons[name] = button
            preset_row.addWidget(button)
        preset_row.addStretch(1)
        self.preset_hint = QLabel("Choose a look, then fine-tune on the right")
        self.preset_hint.setObjectName("subtleText")
        preset_row.addWidget(self.preset_hint)
        layout.addLayout(preset_row)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.undo_button = QPushButton("↶  Undo")
        self.undo_button.setToolTip("Undo the last preset or slider edit (Ctrl+Z).")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button = QPushButton("↷  Redo")
        self.redo_button.setToolTip("Redo the last undone edit (Ctrl+Shift+Z).")
        self.redo_button.clicked.connect(self.redo)
        actions.addWidget(self.undo_button)
        actions.addWidget(self.redo_button)
        actions.addSpacing(4)
        self.before_button = QPushButton("Before")
        self.before_button.setToolTip("Hold to view the exact unedited source image.")
        self.before_button.pressed.connect(self._show_before)
        self.before_button.released.connect(self._show_after)
        self.after_button = QPushButton("After")
        self.after_button.clicked.connect(self._show_after)
        self.split_button = QPushButton("Split")
        self.split_button.setCheckable(True)
        self.split_button.setObjectName("toggleButton")
        self.split_button.setToolTip("Show original left and current preview right.")
        self.split_button.toggled.connect(self._split_toggled)
        actions.addWidget(self.before_button)
        actions.addWidget(self.after_button)
        actions.addWidget(self.split_button)
        actions.addStretch(1)
        self.auto_beauty_button = QPushButton("✦  AUTO BEAUTY")
        self.auto_beauty_button.setObjectName("primaryButton")
        self.auto_beauty_button.setToolTip(
            "Apply the balanced Beauty preset to this photo."
        )
        self.auto_beauty_button.clicked.connect(lambda: self.apply_preset("Beauty"))
        self.export_button = QPushButton("Export Photo")
        self.export_button.setObjectName("exportButton")
        self.export_button.setToolTip(
            "Render at original resolution and save a new file."
        )
        self.export_button.clicked.connect(self.export_current)
        self.batch_button = QPushButton("Process All")
        self.batch_button.setToolTip(
            "Process every queued photo at full resolution with this look."
        )
        self.batch_button.clicked.connect(self.process_all)
        self.watch_button = QPushButton("Watch Folder")
        self.watch_button.setToolTip(
            "Automatically process stable, newly arrived files in a folder."
        )
        self.watch_button.clicked.connect(self.toggle_watch_folder)
        actions.addWidget(self.auto_beauty_button)
        actions.addWidget(self.export_button)
        actions.addWidget(self.batch_button)
        actions.addWidget(self.watch_button)
        layout.addLayout(actions)
        progress_row = QHBoxLayout()
        self.operation_label = QLabel("")
        self.operation_label.setObjectName("subtleText")
        progress_row.addWidget(self.operation_label, 1)
        self.operation_progress = QProgressBar()
        self.operation_progress.setRange(0, 100)
        self.operation_progress.setValue(0)
        self.operation_progress.setMaximumWidth(290)
        self.operation_progress.setVisible(False)
        progress_row.addWidget(self.operation_progress)
        layout.addLayout(progress_row)
        return panel

    def _show_before(self) -> None:
        if self.split_button.isChecked():
            self.split_button.setChecked(False)
        self.canvas.set_mode("before")

    def _show_after(self) -> None:
        if self.split_button.isChecked():
            self.split_button.setChecked(False)
        self.canvas.set_mode("after")

    def _split_toggled(self, enabled: bool) -> None:
        self.canvas.set_mode("split" if enabled else "after")

    def _set_shortcuts(self) -> None:
        shortcuts = (
            ("Ctrl+O", self.open_photos),
            ("Ctrl+S", self.export_current),
            ("Ctrl+Z", self.undo),
            ("Ctrl+Shift+Z", self.redo),
        )
        self._shortcuts: list[QShortcut] = []
        for sequence, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        from PySide6.QtWidgets import QApplication

        QApplication.instance().installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            key_event = event
            if key_event.key() == Qt.Key.Key_Space:
                from PySide6.QtWidgets import QApplication

                focus = QApplication.focusWidget()
                if isinstance(focus, (QLineEdit, QSlider, QSpinBox, QComboBox)):
                    return False
                if (
                    event.type() == QEvent.Type.KeyPress
                    and not key_event.isAutoRepeat()
                ):
                    self._show_before()
                    return True
                if (
                    event.type() == QEvent.Type.KeyRelease
                    and not key_event.isAutoRepeat()
                ):
                    self._show_after()
                    return True
        return super().eventFilter(watched, event)

    def open_photos(self) -> None:
        directory = str(self.settings.get("last_open_directory", str(Path.home())))
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open portrait photos",
            directory,
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)",
        )
        if paths:
            self.add_files(paths)

    def add_files(self, paths: list[str] | tuple[str, ...]) -> None:
        expanded: list[Path] = []
        for raw in paths:
            try:
                path = Path(raw).expanduser()
                if path.is_dir():
                    expanded.extend(
                        sorted(
                            (
                                child
                                for child in path.iterdir()
                                if child.is_file() and is_supported_image(child)
                            ),
                            key=lambda p: p.name.casefold(),
                        )
                    )
                elif path.is_file() and is_supported_image(path):
                    expanded.append(path)
            except OSError as exc:
                logger.warning("Could not inspect dropped path %s: %s", raw, exc)
        added = 0
        first_item = None
        for path in expanded:
            try:
                resolved = path.resolve()
                key = os.path.normcase(str(resolved))
            except OSError:
                resolved = path.absolute()
                key = os.path.normcase(str(resolved))
            if key in self.entries:
                continue
            entry = PhotoEntry(resolved, self.default_state.copy())
            self.entries[key] = entry
            item = QListWidgetItem(QIcon(), resolved.name)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(str(resolved))
            item.setSizeHint(QSize(0, 66))
            self.photo_list.addItem(item)
            if first_item is None:
                first_item = item
            worker = ThumbnailWorker(resolved)
            worker.signals.ready.connect(self._thumbnail_ready)
            self._pool.start(
                worker, -1
            )  # thumbnails yield to opening/rendering the selected source
            added += 1
            logger.info("Photo added to queue: %s", resolved)
        self.photo_count_label.setText(str(len(self.entries)))
        self._set_document_controls(self.current_entry is not None)
        if first_item is not None:
            self.photo_list.setCurrentItem(first_item)
        if expanded:
            self.settings.set("last_open_directory", str(expanded[0].parent))
        if added == 0 and expanded:
            self.statusBar().showMessage("Those photos are already in the queue", 3500)
        elif added:
            self.statusBar().showMessage(
                f"Added {added} photo{'s' if added != 1 else ''} to the queue", 3500
            )

    def _thumbnail_ready(self, path: str, pixels: object, error: object) -> None:
        try:
            resolved_key = os.path.normcase(str(Path(path).resolve()))
        except OSError:
            resolved_key = os.path.normcase(path)
        for index in range(self.photo_list.count()):
            item = self.photo_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) != resolved_key:
                continue
            if pixels is not None:
                qimage = ImageCanvas._qimage(pixels)
                pixmap = QPixmap.fromImage(qimage).scaled(
                    58,
                    52,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                item.setIcon(QIcon(pixmap))
            elif error:
                item.setText(Path(path).name + "  ·  unreadable")
                item.setToolTip(f"{path}\n{error}")
            break

    def _selection_changed(
        self, current: QListWidgetItem | None, previous: QListWidgetItem | None = None
    ) -> None:
        self._commit_history()
        if current is None:
            self.current_entry = None
            self.current_key = None
            self.document = None
            self.preview_source = None
            self.preview_faces = None
            self.last_preview = None
            self.header_file_label.setText("No photo selected")
            self.preview_file_label.setText("Preview")
            self.face_status_label.setText("Add a photo to begin")
            self.canvas.set_images(None, None)
            self._sync_controls()
            self._set_document_controls(False)
            return
        key = current.data(Qt.ItemDataRole.UserRole)
        entry = self.entries.get(key)
        if entry is None:
            return
        self.current_entry = entry
        self.current_key = key
        self.document = None
        self.preview_source = None
        self.preview_faces = None
        self.last_preview = None
        self.header_file_label.setText(entry.path.name)
        self.preview_file_label.setText(entry.path.name)
        self.face_status_label.setText("Loading photo and detecting faces…")
        self.preview_status_label.setText(
            "Loading a preview; the original file is unchanged"
        )
        self.canvas.reset_view()
        self.canvas.set_images(None, None)
        self._sync_controls()
        self._set_document_controls(True)
        worker = LoadWorker(entry.path)
        worker.signals.loaded.connect(self._image_loaded)
        self._pool.start(worker, 100)

    def _image_loaded(
        self, path: str, document: object, preview: object, faces: object, error: object
    ) -> None:
        try:
            loaded_key = os.path.normcase(str(Path(path).resolve()))
        except OSError:
            loaded_key = os.path.normcase(path)
        if loaded_key != self.current_key or self.current_entry is None:
            return
        if error or document is None:
            self.document = None
            self.preview_source = None
            self.face_status_label.setText("Could not read this image")
            self.preview_status_label.setText(str(error))
            self.statusBar().showMessage(
                "Image import failed; technical details are in the local log", 7000
            )
            self._set_document_controls(False)
            logger.error("Image load failed for %s: %s", path, error)
            return
        self.document = document
        self.preview_source = preview
        self.preview_faces = (
            faces if isinstance(faces, FaceAnalysis) else FaceAnalysis()
        )
        self.current_entry.faces = self.preview_faces
        self.canvas.reset_view()
        self.canvas.set_images(self.preview_source, None)
        face_count = len(self.preview_faces.faces)
        if face_count:
            source = (
                "MediaPipe mesh"
                if self.preview_faces.method == "mediapipe-face-mesh"
                else "landmarks estimated"
            )
            self.face_status_label.setText(
                f"{face_count} face{'s' if face_count != 1 else ''} · {source}"
            )
        else:
            self.face_status_label.setText(
                "No face found · color controls are still available"
            )
        self.preview_status_label.setText(
            f"{self.document.width:,} × {self.document.height:,} px · rendering preview…"
        )
        self._set_document_controls(True)
        logger.info(
            "Image loaded: %s (%s x %s), faces=%d",
            path,
            self.document.width,
            self.document.height,
            face_count,
        )
        self._request_preview()

    def _set_document_controls(self, available: bool) -> None:
        for control in getattr(self, "controls", {}).values():
            control.setEnabled(available)
        for button_name in (
            "auto_wb_button",
            "auto_exposure_button",
            "auto_beauty_button",
        ):
            button = getattr(self, button_name, None)
            if button is not None:
                button.setEnabled(available and not self._batch_running)
        if hasattr(self, "export_button"):
            self.export_button.setEnabled(
                self.document is not None
                and not self._batch_running
                and not self._export_running
            )
        if hasattr(self, "batch_button"):
            self.batch_button.setEnabled(
                bool(self.entries)
                and not self._batch_running
                and not self._export_running
            )
        if hasattr(self, "remove_button"):
            self.remove_button.setEnabled(self.photo_list.currentItem() is not None)
        if hasattr(self, "undo_button"):
            self._update_history_buttons()

    def _sync_controls(self) -> None:
        if not hasattr(self, "controls"):
            return
        state = self.current_entry.state if self.current_entry else self.default_state
        self._updating_controls = True
        try:
            for name, control in self.controls.items():
                control.setValue(getattr(state, name))
            if hasattr(self, "auto_wb_button"):
                self.auto_wb_button.setChecked(state.auto_white_balance)
                self.auto_exposure_button.setChecked(state.auto_exposure)
            if hasattr(self, "preset_buttons"):
                button = self.preset_buttons.get(state.preset)
                if button:
                    button.setChecked(True)
        finally:
            self._updating_controls = False
        self._update_history_buttons()

    def _begin_history(self) -> None:
        if self._updating_controls or self.current_entry is None:
            return
        if self._history_before is None:
            self._history_before = self.current_entry.state.to_dict()

    def _control_changed(self, name: str, value: int) -> None:
        if self._updating_controls:
            return
        self._begin_history()
        if self.current_entry is not None:
            setattr(self.current_entry.state, name, value)
            self.current_entry.state.normalize()
            self.default_state = self.current_entry.state.copy()
            self._update_history_buttons()
            self._save_quick_preferences()
            self._request_preview()
            slider = self.controls[name].slider
            if not slider.isSliderDown():
                self._history_timer.start()

    def _commit_history(self) -> None:
        self._history_timer.stop()
        before = self._history_before
        self._history_before = None
        if before is None or self.current_entry is None:
            return
        after = self.current_entry.state.to_dict()
        if before != after:
            self.current_entry.undo_stack.append(before)
            if len(self.current_entry.undo_stack) > 50:
                del self.current_entry.undo_stack[0]
            self.current_entry.redo_stack.clear()
        self._update_history_buttons()

    def _finish_custom_edit(self, mutate) -> None:
        if self.current_entry is None:
            return
        self._begin_history()
        mutate(self.current_entry.state)
        self.current_entry.state.normalize()
        self.default_state = self.current_entry.state.copy()
        self._commit_history()
        self._sync_controls()
        self._save_quick_preferences()
        self._request_preview()

    def apply_preset(self, name: str) -> None:
        if name not in available_presets():
            return
        if self.current_entry is None:
            self.default_state = make_preset(name)
            self.settings.set("last_preset", name)
            self.settings.set("beauty_strength", self.default_state.master_strength)
            if hasattr(self, "preset_buttons"):
                self.preset_buttons[name].setChecked(True)
            self._sync_controls()
            return
        self._begin_history()
        self.current_entry.state = make_preset(name)
        self.default_state = self.current_entry.state.copy()
        self._commit_history()
        self._sync_controls()
        self._save_quick_preferences()
        self._request_preview()
        self.statusBar().showMessage(f"{name} look applied", 2200)

    def _toggle_auto_wb(self, checked: bool) -> None:
        if self.current_entry is not None:
            self._finish_custom_edit(
                lambda state: setattr(state, "auto_white_balance", checked)
            )

    def _toggle_auto_exposure(self, checked: bool) -> None:
        if self.current_entry is not None:
            self._finish_custom_edit(
                lambda state: setattr(state, "auto_exposure", checked)
            )

    def _request_preview(self) -> None:
        if self.preview_source is None or self.current_entry is None:
            return
        self._preview_revision += 1
        self.preview_status_label.setText("Updating preview…")
        self._preview_timer.start()

    def _dispatch_preview(self) -> None:
        if self.preview_source is None or self.current_entry is None:
            return
        if self._preview_inflight:
            self._preview_pending = True
            return
        self._preview_pending = False
        self._preview_inflight = True
        token = self._preview_revision
        worker = PreviewWorker(
            token,
            self.preview_source,
            self.current_entry.state,
            self.preview_faces or FaceAnalysis(),
        )
        worker.signals.rendered.connect(self._preview_ready)
        self._pool.start(worker, 100)

    def _preview_ready(self, token: int, pixels: object, error: object) -> None:
        self._preview_inflight = False
        if token == self._preview_revision and self.preview_source is not None:
            if error:
                self.preview_status_label.setText("Preview failed; see local log")
                self.statusBar().showMessage("Preview could not be rendered", 4000)
            elif pixels is not None:
                self.last_preview = pixels
                self.canvas.set_images(self.preview_source, pixels)
                self.preview_status_label.setText(
                    f"{self.document.width:,} × {self.document.height:,} px · preview only"
                    if self.document
                    else "Preview ready"
                )
        if self._preview_pending or token != self._preview_revision:
            self._preview_pending = False
            self._dispatch_preview()

    def undo(self) -> None:
        self._commit_history()
        if self.current_entry is None or not self.current_entry.undo_stack:
            return
        entry = self.current_entry
        entry.redo_stack.append(entry.state.to_dict())
        entry.state = EditState.from_dict(entry.undo_stack.pop())
        self.default_state = entry.state.copy()
        self._sync_controls()
        self._save_quick_preferences()
        self._request_preview()

    def redo(self) -> None:
        self._commit_history()
        if self.current_entry is None or not self.current_entry.redo_stack:
            return
        entry = self.current_entry
        entry.undo_stack.append(entry.state.to_dict())
        entry.state = EditState.from_dict(entry.redo_stack.pop())
        self.default_state = entry.state.copy()
        self._sync_controls()
        self._save_quick_preferences()
        self._request_preview()

    def _update_history_buttons(self) -> None:
        if not hasattr(self, "undo_button"):
            return
        self.undo_button.setEnabled(
            bool(self.current_entry and self.current_entry.undo_stack)
        )
        self.redo_button.setEnabled(
            bool(self.current_entry and self.current_entry.redo_stack)
        )

    def _save_quick_preferences(self) -> None:
        state = self.current_entry.state if self.current_entry else self.default_state
        self.settings.set("last_preset", state.preset)
        self.settings.set("beauty_strength", state.master_strength)

    def _output_settings_changed(self, *args) -> None:
        self.settings.update(
            {
                "last_output_directory": self.output_folder_edit.text().strip(),
                "filename_suffix": self.suffix_edit.text().strip(),
                "jpeg_quality": self.quality_spin.value(),
            }
        )
        if hasattr(self, "format_combo"):
            self.settings.set("output_format", self.format_combo.currentData())

    def _browse_output_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose output folder",
            self.output_folder_edit.text() or str(Path.home()),
        )
        if path:
            self.output_folder_edit.setText(path)
            self._output_settings_changed()

    def _output_config(self) -> tuple[Path, str, str, int]:
        folder_text = self.output_folder_edit.text().strip()
        folder = (
            Path(folder_text).expanduser()
            if folder_text
            else Path.home() / "Pictures" / "GlowUp"
        )
        suffix = self.suffix_edit.text().strip() or "beauty"
        extension = self.format_combo.currentData() or ".jpg"
        quality = self.quality_spin.value()
        return folder, suffix, extension, quality

    def export_current(self) -> None:
        if (
            self.document is None
            or self.current_entry is None
            or self._export_running
            or self._batch_running
        ):
            return
        folder, suffix, extension, quality = self._output_config()
        try:
            destination = output_path_for(self.document.path, folder, suffix, extension)
            self._export_running = True
            self.export_button.setEnabled(False)
            self.batch_button.setEnabled(False)
            self.operation_label.setText("Rendering full-resolution export…")
            self.operation_progress.setVisible(True)
            self.operation_progress.setRange(0, 0)
            worker = ExportWorker(
                self.document, self.current_entry.state, destination, quality
            )
            worker.signals.finished.connect(self._export_finished)
            self._pool.start(worker)
            logger.info("Full-resolution export started: %s", destination)
        except Exception as exc:
            logger.exception("Could not begin export")
            self.statusBar().showMessage(f"Export could not start: {exc}", 7000)

    def _export_finished(self, path: object, error: object) -> None:
        self._export_running = False
        self.operation_progress.setVisible(False)
        self.operation_progress.setRange(0, 100)
        self.export_button.setEnabled(
            self.document is not None and not self._batch_running
        )
        self.batch_button.setEnabled(not self._batch_running)
        self._set_document_controls(self.document is not None)
        if error:
            self.operation_label.setText("Export failed")
            self.statusBar().showMessage(f"Export failed: {error}", 8000)
        else:
            self.operation_label.setText(f"Saved: {path}")
            self.statusBar().showMessage(f"Export saved · {path}", 6000)

    def process_all(self) -> None:
        if self._batch_running or self._export_running or not self.entries:
            return
        folder, suffix, extension, quality = self._output_config()
        state = (
            self.current_entry.state.copy()
            if self.current_entry
            else self.default_state.copy()
        )
        files = [str(entry.path) for entry in self.entries.values()]
        self._batch_running = True
        self.batch_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.operation_progress.setVisible(True)
        self.operation_progress.setRange(0, len(files))
        self.operation_progress.setValue(0)
        self.operation_label.setText(f"Processing {len(files)} photos…")
        worker = BatchWorker(files, state, folder, quality, suffix, extension)
        worker.signals.progress.connect(self._batch_progress)
        worker.signals.finished.connect(self._batch_finished)
        self._pool.start(worker)
        logger.info("Batch processing started: %d photos", len(files))

    def _batch_progress(self, done: int, total: int, item: BatchResult) -> None:
        self.operation_progress.setRange(0, max(1, total))
        self.operation_progress.setValue(done)
        status = f"Processed {done} of {total} · {item.source.name}"
        if not item.succeeded:
            status += f" (failed: {item.error})"
        self.operation_label.setText(status)

    def _batch_finished(self, results: object) -> None:
        self._batch_running = False
        self.operation_progress.setVisible(False)
        self.batch_button.setEnabled(bool(self.entries))
        self.export_button.setEnabled(
            self.document is not None and not self._export_running
        )
        if isinstance(results, list) and results and isinstance(results[0], Exception):
            self.operation_label.setText("Batch stopped")
            self.statusBar().showMessage(f"Batch failed: {results[0]}", 8000)
            return
        results = results if isinstance(results, list) else []
        succeeded = sum(
            1
            for result in results
            if isinstance(result, BatchResult) and result.succeeded
        )
        failed = len(results) - succeeded
        self.operation_label.setText(
            f"Batch complete · {succeeded} exported · {failed} failed"
        )
        self.statusBar().showMessage(
            f"Batch complete: {succeeded} exported, {failed} failed", 7000
        )
        logger.info("Batch finished: %d succeeded, %d failed", succeeded, failed)

    def toggle_watch_folder(self) -> None:
        thread = self._watcher._thread
        if thread is not None and thread.is_alive():
            self._watcher.stop()
            self.watch_button.setText("Stopping…")
            self.watch_button.setEnabled(False)
            return
        dialog = WatchFolderDialog(
            str(self.settings.get("watch_input_folder", "")),
            str(
                self.settings.get("watch_output_folder", self.output_folder_edit.text())
            ),
            self,
        )
        if dialog.exec() != WatchFolderDialog.DialogCode.Accepted:
            return
        input_folder = dialog.input_edit.text().strip()
        output_folder = dialog.output_edit.text().strip()
        state = (
            self.current_entry.state.copy()
            if self.current_entry
            else self.default_state.copy()
        )
        _, suffix, extension, quality = self._output_config()
        try:
            self._watcher.start(
                input_folder,
                output_folder,
                state,
                quality=quality,
                suffix=suffix,
                extension=extension,
                on_status=self._bridge.watch_status.emit,
                on_item=self._bridge.watch_item.emit,
            )
            self.settings.update(
                {
                    "watch_input_folder": input_folder,
                    "watch_output_folder": output_folder,
                }
            )
            self.watch_button.setText("Stop Watching")
            self.watch_button.setEnabled(True)
            self.operation_label.setText("Watch Folder is running")
        except Exception as exc:
            logger.exception("Could not start watch folder")
            QMessageBox.warning(
                self, "Watch Folder", f"Could not start watching:\n{exc}"
            )

    def _on_watch_status(self, text: str) -> None:
        self.operation_label.setText(text)
        if text == "Stopped":
            self.watch_button.setText("Watch Folder")
            self.watch_button.setEnabled(True)
        elif text.startswith("Stopping"):
            self.watch_button.setText("Stopping…")
            self.watch_button.setEnabled(False)
        elif text.startswith("Watching"):
            self.watch_button.setText("Stop Watching")
            self.watch_button.setEnabled(True)

    def _on_watch_item(self, result: BatchResult) -> None:
        if result.succeeded:
            self.statusBar().showMessage(f"Watch Folder saved · {result.output}", 4500)
        else:
            self.statusBar().showMessage(
                f"Watch Folder could not process {result.source.name}: {result.error}",
                6500,
            )

    def remove_selected(self) -> None:
        item = self.photo_list.currentItem()
        if item is None:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        was_current = key == self.current_key
        row = self.photo_list.row(item)
        self.photo_list.takeItem(row)
        self.entries.pop(key, None)
        self.photo_count_label.setText(str(len(self.entries)))
        if was_current:
            self.current_entry = None
            self.current_key = None
            self.document = None
            self.preview_source = None
            self.preview_faces = None
            self.last_preview = None
            if self.photo_list.count():
                self.photo_list.setCurrentRow(min(row, self.photo_list.count() - 1))
            else:
                self._selection_changed(None)
        logger.info("Photo removed from queue: %s", key)

    def closeEvent(self, event) -> None:
        from PySide6.QtWidgets import QApplication

        self._history_timer.stop()
        self._commit_history()
        self._preview_timer.stop()
        self._watcher.detach_callbacks()
        self._watcher.stop(wait=False)
        QApplication.instance().removeEventFilter(self)
        self._save_quick_preferences()
        try:
            self.settings.save()
        except OSError:
            logger.exception("Could not save settings during shutdown")
        logger.info("GlowUp Local closing")
        # Do not destroy Qt worker signal owners while native processing is still running.
        self._pool.clear()
        self._pool.waitForDone()
        self._watcher.stop(wait=True, timeout=5.0)
        super().closeEvent(event)
