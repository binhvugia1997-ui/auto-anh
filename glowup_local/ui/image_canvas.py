"""Zoomable, pannable before/after image canvas."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


class ImageCanvas(QWidget):
    files_dropped = Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 260)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.original = QImage()
        self.edited = QImage()
        self.mode = "after"  # after, before, or split
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._drag_pos: QPoint | None = None
        self.setStyleSheet("background:#11151b;")

    @staticmethod
    def _qimage(image: np.ndarray) -> QImage:
        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        if image.ndim == 2:
            qimage = QImage(
                image.data,
                width,
                height,
                image.strides[0],
                QImage.Format.Format_Grayscale8,
            )
        elif image.shape[2] == 4:
            qimage = QImage(
                image.data,
                width,
                height,
                image.strides[0],
                QImage.Format.Format_RGBA8888,
            )
        else:
            qimage = QImage(
                image.data, width, height, image.strides[0], QImage.Format.Format_RGB888
            )
        return qimage.copy()  # detach from NumPy worker buffers

    def set_images(
        self, original: np.ndarray | None, edited: np.ndarray | None
    ) -> None:
        self.original = self._qimage(original) if original is not None else QImage()
        self.edited = self._qimage(edited) if edited is not None else QImage()
        self.update()

    def set_mode(self, mode: str) -> None:
        if mode not in {"after", "before", "split"}:
            raise ValueError(mode)
        self.mode = mode
        self.update()

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.update()

    def _image_rect(self) -> QRectF:
        image = (
            self.original
            if self.mode == "before" or self.edited.isNull()
            else self.edited
        )
        if image.isNull() or self.width() < 1 or self.height() < 1:
            return QRectF()
        fit = min(
            (self.width() - 36) / image.width(), (self.height() - 36) / image.height()
        )
        scale = max(0.01, fit) * self.zoom
        width, height = image.width() * scale, image.height() * scale
        center_x = self.width() / 2 + self.pan.x()
        center_y = self.height() / 2 + self.pan.y()
        return QRectF(center_x - width / 2, center_y - height / 2, width, height)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#11151b"))
        if self.original.isNull():
            painter.setPen(QColor("#a9b2bf"))
            title_font = QFont(self.font())
            title_font.setPointSize(15)
            title_font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(title_font)
            painter.drawText(
                self.rect().adjusted(30, 0, -30, -35),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                "Drop a portrait photo here\n\nor choose  Open Photos",
            )
            painter.setPen(QColor("#737f8e"))
            painter.setFont(QFont(self.font().family(), 9))
            painter.drawText(
                self.rect().adjusted(20, 0, -20, -65),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                "JPG · PNG · WEBP  |  Your photos stay on this device",
            )
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = self._image_rect()
        if self.mode == "before" or self.edited.isNull():
            painter.drawImage(rect, self.original)
            self._draw_tag(painter, "BEFORE", 16)
        elif self.mode == "split":
            split_x = self.width() / 2
            painter.save()
            painter.setClipRect(QRectF(0, 0, split_x, self.height()))
            painter.drawImage(rect, self.original)
            painter.restore()
            painter.save()
            painter.setClipRect(
                QRectF(split_x, 0, self.width() - split_x, self.height())
            )
            painter.drawImage(rect, self.edited)
            painter.restore()
            painter.setPen(QPen(QColor("#f3f5f8"), 1.5))
            painter.drawLine(int(split_x), 0, int(split_x), self.height())
            self._draw_tag(painter, "BEFORE", 16)
            self._draw_tag(painter, "AFTER", max(16, int(split_x) + 16))
        else:
            painter.drawImage(rect, self.edited)
            self._draw_tag(painter, "AFTER", 16)
        painter.setPen(QColor(255, 255, 255, 145))
        painter.setFont(QFont(self.font().family(), 9))
        painter.drawText(
            self.rect().adjusted(0, 0, -14, -10),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            f"{round(self.zoom * 100)}%",
        )

    @staticmethod
    def _draw_tag(painter: QPainter, text: str, x: int) -> None:
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(9, 13, 18, 190))
        painter.drawRoundedRect(QRectF(x, 15, 67, 25), 6, 6)
        painter.setPen(QColor("#f4f6f9"))
        painter.setFont(QFont(painter.font().family(), 8, QFont.Weight.Bold))
        painter.drawText(QRectF(x, 15, 67, 25), Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def wheelEvent(self, event) -> None:
        if self.original.isNull():
            return
        old = self.zoom
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = min(6.0, max(0.25, self.zoom * factor))
        if old != self.zoom:
            position = event.position()
            center = QPointF(self.width() / 2, self.height() / 2)
            relative = position - center - self.pan
            ratio = self.zoom / old
            self.pan = self.pan - relative * (ratio - 1.0)
            self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.reset_view()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.zoom > 1.0:
            self._drag_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos is not None:
            current = event.position().toPoint()
            delta = current - self._drag_pos
            self.pan += QPointF(delta)
            self._drag_pos = current
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        paths = [
            url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()
        ]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()
