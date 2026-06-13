import cv2
import numpy as np
from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

from core.config import DEPTH_VMIN_MM, DEPTH_VMAX_MM


def ndarray_to_pixmap(img_rgb: np.ndarray) -> QPixmap:
    """Convert an RGB numpy array to QPixmap. Copies buffer to avoid dangling pointer."""
    img = np.ascontiguousarray(img_rgb)
    h, w, c = img.shape
    qimg = QImage(img.data, w, h, w * c, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def depth_to_rgb(depth_u16: np.ndarray) -> np.ndarray:
    """Convert uint16 depth map to RGB pseudo-color."""
    d = depth_u16.astype(np.float32)
    valid = d > 0
    d = np.clip(d, DEPTH_VMIN_MM, DEPTH_VMAX_MM)
    d_norm = (d - DEPTH_VMIN_MM) / (DEPTH_VMAX_MM - DEPTH_VMIN_MM + 1e-6)
    d_u8 = (d_norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


class ImageLabel(QLabel):
    """QLabel that auto-scales images while keeping aspect ratio."""

    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setStyleSheet(
            "background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px; color: #9e9e9e;"
        )
        self._pixmap_src = None
        if placeholder:
            self.setText(placeholder)

    def set_image(self, img_rgb: np.ndarray):
        self._pixmap_src = ndarray_to_pixmap(img_rgb)
        self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._pixmap_src is None:
            return
        scaled = self._pixmap_src.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)
