import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QPushButton, QGridLayout,
)
from PyQt6.QtCore import pyqtSignal as Signal, Qt

from .utils import ImageLabel, ndarray_to_pixmap

_BTN_STYLE = """
    QPushButton {
        background: #1976d2; color: white;
        border-radius: 6px; font-size: 13px;
        padding: 8px;
    }
    QPushButton:hover   { background: #1565c0; }
    QPushButton:pressed { background: #0d47a1; }
"""

METRIC_NAMES = [
    "张口度", "甲颌距离", "头后仰角度", "下颌活动角度",
    "下颌活动度", "颈围", "舌厚度", "声门视野",
]


class ResultWidget(QWidget):
    """
    Page 3: closed/open mesh projections + metric list + restart button.

    Signal restart is emitted when the user clicks the restart button.
    """

    restart = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._metric_labels: dict[str, QLabel] = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # left: 2x2 grid (RGB with ArUco + mesh renders)
        grid = QGridLayout()
        grid.setSpacing(8)

        for col, title in enumerate(["原图 (ArUco)", "拟合网格"]):
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #9e9e9e; font-size: 11px;")
            grid.addWidget(lbl, 0, col)

        self.img_rgb_closed  = ImageLabel("闭口 RGB")
        self.img_closed      = ImageLabel("闭口 Mesh")
        self.img_rgb_open    = ImageLabel("张口 RGB")
        self.img_open        = ImageLabel("张口 Mesh")

        grid.addWidget(self.img_rgb_closed, 1, 0)
        grid.addWidget(self.img_closed,     1, 1)
        grid.addWidget(self.img_rgb_open,   2, 0)
        grid.addWidget(self.img_open,       2, 1)

        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 1)

        grid_wrap = QWidget()
        grid_wrap.setLayout(grid)
        root.addWidget(grid_wrap, stretch=3)

        # right: metrics + button
        sidebar = QFrame()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("background: #ffffff; border: none; border-radius: 8px;")
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(20, 24, 20, 24)
        sb.setSpacing(8)

        lbl_title = QLabel("评估结果")
        lbl_title.setStyleSheet(
            "color: #212121; font-size: 16px; font-weight: bold;"
        )
        sb.addWidget(lbl_title)
        sb.addSpacing(8)

        # metric rows
        for name in METRIC_NAMES:
            row = QHBoxLayout()
            lbl_name = QLabel(f"{name}:")
            lbl_name.setStyleSheet("color: #616161; font-size: 13px;")
            lbl_val = QLabel("--")
            lbl_val.setStyleSheet(
                "color: #1976d2; font-size: 13px; font-weight: bold;"
            )
            lbl_val.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            row.addWidget(lbl_name)
            row.addWidget(lbl_val)
            sb.addLayout(row)
            self._metric_labels[name] = lbl_val

        sb.addStretch()

        btn = QPushButton("重新采集")
        btn.setFixedHeight(40)
        btn.setStyleSheet(_BTN_STYLE)
        btn.clicked.connect(self.restart.emit)
        sb.addWidget(btn)
        root.addWidget(sidebar)

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------
    def setup(self, metrics: dict, proj_closed: np.ndarray, proj_open: np.ndarray,
              rgb_closed: np.ndarray = None, rgb_open: np.ndarray = None,
              aruco_corners: dict = None):
        """
        proj_closed / proj_open: BGR numpy arrays returned by auto_render_mesh.
        rgb_closed / rgb_open: original RGB captures.
        aruco_corners: {"closed": corners_2d, "open": corners_2d} or None.
        """
        self.img_closed.set_image(cv2.cvtColor(proj_closed, cv2.COLOR_BGR2RGB))
        self.img_open.set_image(cv2.cvtColor(proj_open,   cv2.COLOR_BGR2RGB))

        if rgb_closed is not None:
            vis_c = self._overlay_aruco(rgb_closed,
                                        aruco_corners.get("closed") if aruco_corners else None)
            self.img_rgb_closed.set_image(vis_c)
        if rgb_open is not None:
            vis_o = self._overlay_aruco(rgb_open,
                                        aruco_corners.get("open") if aruco_corners else None)
            self.img_rgb_open.set_image(vis_o)

        for name, val in metrics.items():
            if name in self._metric_labels:
                self._metric_labels[name].setText(val)

    @staticmethod
    def _overlay_aruco(rgb: np.ndarray, corners_2d) -> np.ndarray:
        vis = rgb.copy()
        if corners_2d is None:
            return vis
        pts = corners_2d.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        center = corners_2d.mean(axis=0).astype(np.int32)
        cv2.circle(vis, (center[0], center[1]), 4, (0, 0, 255), -1)
        return vis
