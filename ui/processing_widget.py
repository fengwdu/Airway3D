import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QGridLayout,
)
from PyQt6.QtCore import Qt

from .utils import ImageLabel, depth_to_rgb

_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
_ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)


class ProcessingWidget(QWidget):
    """
    Page 2: four static captured images + live step status on the right.

    Call setup() to initialize images, set_status() to update progress text.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # 四宫格
        grid = QGridLayout()
        grid.setSpacing(8)
        self.img_c_rgb   = ImageLabel("闭口 RGB")
        self.img_c_depth = ImageLabel("闭口 Depth")
        self.img_o_rgb   = ImageLabel("张口 RGB")
        self.img_o_depth = ImageLabel("张口 Depth")
        grid.addWidget(self.img_c_rgb,   0, 0)
        grid.addWidget(self.img_c_depth, 0, 1)
        grid.addWidget(self.img_o_rgb,   1, 0)
        grid.addWidget(self.img_o_depth, 1, 1)
        grid_wrap = QWidget()
        grid_wrap.setLayout(grid)
        root.addWidget(grid_wrap, stretch=3)

        # right: status sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("background: #ffffff; border: none; border-radius: 8px;")
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(20, 30, 20, 30)
        sb.setSpacing(12)

        lbl_title = QLabel("正在处理...")
        lbl_title.setStyleSheet(
            "color: #212121; font-size: 16px; font-weight: bold;"
        )

        self.lbl_step = QLabel("请稍候")
        self.lbl_step.setStyleSheet("color: #616161; font-size: 13px;")
        self.lbl_step.setWordWrap(True)
        self.lbl_step.setAlignment(Qt.AlignmentFlag.AlignTop)

        sb.addWidget(lbl_title)
        sb.addSpacing(8)
        sb.addWidget(self.lbl_step)
        sb.addStretch()
        root.addWidget(sidebar)

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------
    def setup(self, rgb_closed: np.ndarray, rgb_open: np.ndarray,
              depth_closed: np.ndarray, depth_open: np.ndarray):
        self._rgb_closed = rgb_closed
        self._rgb_open   = rgb_open
        # ArUco 仅在闭口图像上执行
        self.img_c_rgb.set_image(self._overlay_aruco(rgb_closed))
        self.img_c_depth.set_image(depth_to_rgb(depth_closed))
        self.img_o_rgb.set_image(rgb_open)
        self.img_o_depth.set_image(depth_to_rgb(depth_open))
        self.lbl_step.setText("请稍候")

    @staticmethod
    def _overlay_aruco(rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners_list, ids, _ = _ARUCO_DETECTOR.detectMarkers(gray)
        if ids is None:
            return rgb
        vis = rgb.copy()
        for corners, mid in zip(corners_list, ids):
            pts = corners[0].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            center = corners[0].mean(axis=0).astype(np.int32)
            cv2.circle(vis, (center[0], center[1]), 4, (0, 0, 255), -1)
            cv2.putText(vis, f"ID:{mid[0]}", (center[0] + 8, center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return vis

    def set_status(self, text: str):
        self.lbl_step.setText(text)

    def draw_aruco(self, aruco_corners: dict):
        """ArUco 检测完成后, 在 RGB 图上叠加绿色标记框."""
        from core.aruco import draw_aruco_on_rgb

        corners_c = aruco_corners.get("closed")
        corners_o = aruco_corners.get("open")

        self.img_c_rgb.set_image(draw_aruco_on_rgb(self._rgb_closed, corners_c))
        self.img_o_rgb.set_image(draw_aruco_on_rgb(self._rgb_open, corners_o))
