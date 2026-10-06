"""A small local-only input/output folder chooser for Watch Folder."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class WatchFolderDialog(QDialog):
    def __init__(
        self, input_folder: str = "", output_folder: str = "", parent=None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Watch Folder")
        self.setMinimumWidth(540)
        self.input_edit = QLineEdit(input_folder)
        self.output_edit = QLineEdit(output_folder)
        self.input_edit.setPlaceholderText("Folder where new photos will arrive")
        self.output_edit.setPlaceholderText("Folder for beautified exports")
        form = QFormLayout()
        form.setSpacing(12)
        form.addRow("Input folder", self._row(self.input_edit, self._browse_input))
        form.addRow("Output folder", self._row(self.output_edit, self._browse_output))
        note = QLabel(
            "GlowUp waits for each file to finish writing before it processes it.\n"
            "Existing photos are left alone; new arrivals are processed locally."
        )
        note.setObjectName("subtleText")
        note.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start Watching")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)
        self.error_label = QLabel("")
        self.error_label.setObjectName("errorText")
        layout.insertWidget(1, self.error_label)

    @staticmethod
    def _row(edit: QLineEdit, callback) -> object:
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(callback)
        row.addWidget(button)
        return row

    def _browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose input folder", self.input_edit.text()
        )
        if path:
            self.input_edit.setText(path)
            if not self.output_edit.text().strip():
                self.output_edit.setText(str(Path(path).parent / "GlowUp exports"))

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.output_edit.text()
        )
        if path:
            self.output_edit.setText(path)

    def _validate_and_accept(self) -> None:
        input_path, output_path = (
            self.input_edit.text().strip(),
            self.output_edit.text().strip(),
        )
        if not input_path or not Path(input_path).is_dir():
            self.error_label.setText("Choose an existing input folder.")
            return
        if not output_path:
            self.error_label.setText("Choose an output folder.")
            return
        self.accept()
