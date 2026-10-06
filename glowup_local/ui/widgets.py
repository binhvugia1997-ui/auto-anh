"""Reusable dark-theme Qt controls."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget


class SectionHeader(QLabel):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(title.upper(), parent)
        self.setObjectName("sectionHeader")


class SliderControl(QWidget):
    valueChanged = Signal(int)
    sliderPressed = Signal()
    sliderReleased = Signal()

    def __init__(
        self,
        label: str,
        minimum: int = 0,
        maximum: int = 100,
        value: int = 0,
        tooltip: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum
        self.title = QLabel(label)
        self.title.setObjectName("controlLabel")
        self.value_label = QLabel()
        self.value_label.setObjectName("controlValue")
        self.value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.title, 1)
        top.addWidget(self.value_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, (maximum - minimum) // 10))
        self.slider.setValue(value)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 1)
        layout.setSpacing(0)
        layout.addLayout(top)
        layout.addWidget(self.slider)
        self.slider.valueChanged.connect(self._on_value)
        self.slider.sliderPressed.connect(self.sliderPressed.emit)
        self.slider.sliderReleased.connect(self.sliderReleased.emit)
        self.slider.valueChanged.connect(self.valueChanged.emit)
        if tooltip:
            self.setToolTip(tooltip)
            self.slider.setToolTip(tooltip)
        self._on_value(value)

    def _on_value(self, value: int) -> None:
        self.value_label.setText(f"{value:+d}" if self.minimum < 0 else str(value))

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:
        self.slider.setValue(min(self.maximum, max(self.minimum, int(value))))

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.title.setEnabled(enabled)
        self.value_label.setEnabled(enabled)
        self.slider.setEnabled(enabled)
